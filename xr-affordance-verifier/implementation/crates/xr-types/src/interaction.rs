//! Interaction state machine types.
//!
//! Models multi-step interactions as finite state machines with
//! semialgebraic pose guards. Each [`InteractionStateMachine`] defines
//! states, transitions, and guard predicates that must be satisfied
//! for a transition to fire.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::scene::InteractionType;
use crate::traits::SmtExpr;
use crate::ElementId;

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

/// A named state in an interaction state machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionState {
    /// Unique state identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Whether this is an initial state.
    pub is_initial: bool,
    /// Whether this is an accepting (final) state.
    pub is_accepting: bool,
    /// Interaction type associated with this state (if any).
    pub interaction_type: Option<InteractionType>,
    /// Maximum dwell time in this state (seconds), 0 = unlimited.
    pub max_dwell_s: f64,
}

impl InteractionState {
    /// Create a new non-initial, non-accepting state.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            is_initial: false,
            is_accepting: false,
            interaction_type: None,
            max_dwell_s: 0.0,
        }
    }

    /// Mark this state as initial.
    pub fn initial(mut self) -> Self {
        self.is_initial = true;
        self
    }

    /// Mark this state as accepting.
    pub fn accepting(mut self) -> Self {
        self.is_accepting = true;
        self
    }

    /// Set the interaction type.
    pub fn with_interaction(mut self, it: InteractionType) -> Self {
        self.interaction_type = Some(it);
        self
    }

    /// Set max dwell time.
    pub fn with_max_dwell(mut self, seconds: f64) -> Self {
        self.max_dwell_s = seconds;
        self
    }
}

impl std::fmt::Display for InteractionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut flags = String::new();
        if self.is_initial {
            flags.push_str("→");
        }
        if self.is_accepting {
            flags.push_str("✓");
        }
        write!(f, "{}[{}]{}", self.id, self.label, flags)
    }
}

// ---------------------------------------------------------------------------
// Pose guards
// ---------------------------------------------------------------------------

/// A semialgebraic pose guard predicate.
///
/// Guards are boolean conditions over body parameters and joint angles
/// that must hold for a transition to fire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseGuard {
    /// Human-readable description.
    pub description: String,
    /// The guard predicate type.
    pub predicate: GuardPredicate,
}

/// A concrete guard predicate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GuardPredicate {
    /// End-effector must be within distance `radius` of `center`.
    ProximityBall {
        center: [f64; 3],
        radius: f64,
    },
    /// End-effector must be inside the axis-aligned box.
    InsideBox {
        min: [f64; 3],
        max: [f64; 3],
    },
    /// A joint angle must be within a range.
    JointRange {
        joint_index: usize,
        min_angle: f64,
        max_angle: f64,
    },
    /// Conjunction of sub-guards.
    And(Vec<GuardPredicate>),
    /// Disjunction of sub-guards.
    Or(Vec<GuardPredicate>),
    /// Negation.
    Not(Box<GuardPredicate>),
    /// Always true (no constraint).
    True,
    /// Custom SMT expression.
    Custom(SmtExpr),
}

impl PoseGuard {
    /// Create a trivially-true guard.
    pub fn trivial() -> Self {
        Self {
            description: "always".into(),
            predicate: GuardPredicate::True,
        }
    }

    /// Create a proximity-ball guard.
    pub fn proximity(center: [f64; 3], radius: f64) -> Self {
        Self {
            description: format!("within {radius:.3}m of ({:.2},{:.2},{:.2})", center[0], center[1], center[2]),
            predicate: GuardPredicate::ProximityBall { center, radius },
        }
    }

    /// Create a box guard.
    pub fn inside_box(min: [f64; 3], max: [f64; 3]) -> Self {
        Self {
            description: format!(
                "inside box [{:.2},{:.2},{:.2}]-[{:.2},{:.2},{:.2}]",
                min[0], min[1], min[2], max[0], max[1], max[2]
            ),
            predicate: GuardPredicate::InsideBox { min, max },
        }
    }

    /// Create a joint-range guard.
    pub fn joint_range(joint_index: usize, min_angle: f64, max_angle: f64) -> Self {
        Self {
            description: format!("joint[{joint_index}] in [{min_angle:.2}, {max_angle:.2}]"),
            predicate: GuardPredicate::JointRange {
                joint_index,
                min_angle,
                max_angle,
            },
        }
    }

    /// Create a conjunction of guards.
    pub fn and(guards: Vec<PoseGuard>) -> Self {
        let preds: Vec<GuardPredicate> = guards.into_iter().map(|g| g.predicate).collect();
        Self {
            description: "conjunction".into(),
            predicate: GuardPredicate::And(preds),
        }
    }

    /// Evaluate the guard at a given position and joint angles.
    pub fn evaluate(&self, position: &[f64; 3], joint_angles: &[f64]) -> bool {
        evaluate_predicate(&self.predicate, position, joint_angles)
    }

    /// Encode the guard as an SMT expression.
    pub fn to_smt(&self, pos_vars: &[String; 3], joint_vars: &[String]) -> SmtExpr {
        encode_predicate_smt(&self.predicate, pos_vars, joint_vars)
    }
}

fn evaluate_predicate(pred: &GuardPredicate, pos: &[f64; 3], joints: &[f64]) -> bool {
    match pred {
        GuardPredicate::True => true,
        GuardPredicate::ProximityBall { center, radius } => {
            let dx = pos[0] - center[0];
            let dy = pos[1] - center[1];
            let dz = pos[2] - center[2];
            (dx * dx + dy * dy + dz * dz).sqrt() <= *radius
        }
        GuardPredicate::InsideBox { min, max } => {
            (0..3).all(|i| pos[i] >= min[i] && pos[i] <= max[i])
        }
        GuardPredicate::JointRange {
            joint_index,
            min_angle,
            max_angle,
        } => {
            if let Some(&angle) = joints.get(*joint_index) {
                angle >= *min_angle && angle <= *max_angle
            } else {
                false
            }
        }
        GuardPredicate::And(subs) => subs.iter().all(|s| evaluate_predicate(s, pos, joints)),
        GuardPredicate::Or(subs) => subs.iter().any(|s| evaluate_predicate(s, pos, joints)),
        GuardPredicate::Not(inner) => !evaluate_predicate(inner, pos, joints),
        GuardPredicate::Custom(_) => {
            // Custom guards cannot be evaluated concretely.
            true
        }
    }
}

fn encode_predicate_smt(
    pred: &GuardPredicate,
    pos_vars: &[String; 3],
    joint_vars: &[String],
) -> SmtExpr {
    match pred {
        GuardPredicate::True => SmtExpr::bool_lit(true),
        GuardPredicate::ProximityBall { center, radius } => {
            // (x - cx)^2 + (y - cy)^2 + (z - cz)^2 <= r^2
            let terms: Vec<SmtExpr> = (0..3)
                .map(|i| {
                    let diff = SmtExpr::sub(
                        SmtExpr::sym(&pos_vars[i]),
                        SmtExpr::num(center[i]),
                    );
                    SmtExpr::mul(diff.clone(), diff)
                })
                .collect();
            let sum = terms
                .into_iter()
                .reduce(SmtExpr::add)
                .unwrap_or(SmtExpr::num(0.0));
            SmtExpr::le(sum, SmtExpr::num(radius * radius))
        }
        GuardPredicate::InsideBox { min, max } => {
            let mut constraints = Vec::new();
            for i in 0..3 {
                constraints.push(SmtExpr::le(SmtExpr::num(min[i]), SmtExpr::sym(&pos_vars[i])));
                constraints.push(SmtExpr::le(SmtExpr::sym(&pos_vars[i]), SmtExpr::num(max[i])));
            }
            SmtExpr::and(constraints)
        }
        GuardPredicate::JointRange {
            joint_index,
            min_angle,
            max_angle,
        } => {
            if let Some(var) = joint_vars.get(*joint_index) {
                SmtExpr::and(vec![
                    SmtExpr::le(SmtExpr::num(*min_angle), SmtExpr::sym(var)),
                    SmtExpr::le(SmtExpr::sym(var), SmtExpr::num(*max_angle)),
                ])
            } else {
                SmtExpr::bool_lit(false)
            }
        }
        GuardPredicate::And(subs) => {
            let exprs: Vec<SmtExpr> = subs
                .iter()
                .map(|s| encode_predicate_smt(s, pos_vars, joint_vars))
                .collect();
            SmtExpr::and(exprs)
        }
        GuardPredicate::Or(subs) => {
            let exprs: Vec<SmtExpr> = subs
                .iter()
                .map(|s| encode_predicate_smt(s, pos_vars, joint_vars))
                .collect();
            SmtExpr::or(exprs)
        }
        GuardPredicate::Not(inner) => {
            SmtExpr::not(encode_predicate_smt(inner, pos_vars, joint_vars))
        }
        GuardPredicate::Custom(expr) => expr.clone(),
    }
}

// ---------------------------------------------------------------------------
// Transitions
// ---------------------------------------------------------------------------

/// A transition between two states in the FSM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateTransition {
    /// Unique transition identifier.
    pub id: String,
    /// Source state ID.
    pub from: String,
    /// Target state ID.
    pub to: String,
    /// Guard that must be satisfied.
    pub guard: PoseGuard,
    /// Required interaction type to trigger this transition.
    pub trigger: Option<InteractionType>,
    /// Transition priority (lower = higher priority).
    pub priority: u32,
    /// Human-readable label.
    pub label: String,
}

impl StateTransition {
    /// Create a new transition.
    pub fn new(
        id: impl Into<String>,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            from: from.into(),
            to: to.into(),
            guard: PoseGuard::trivial(),
            trigger: None,
            priority: 0,
            label: String::new(),
        }
    }

    /// Set the guard.
    pub fn with_guard(mut self, guard: PoseGuard) -> Self {
        self.guard = guard;
        self
    }

    /// Set the trigger interaction type.
    pub fn with_trigger(mut self, trigger: InteractionType) -> Self {
        self.trigger = Some(trigger);
        self
    }

    /// Set the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

impl std::fmt::Display for StateTransition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} --[{}]--> {}", self.from, self.label, self.to)
    }
}

// ---------------------------------------------------------------------------
// InteractionStateMachine
// ---------------------------------------------------------------------------

/// A finite state machine modeling a multi-step interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionStateMachine {
    /// Machine identifier.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Element this FSM pertains to.
    pub element_id: ElementId,
    /// States.
    pub states: Vec<InteractionState>,
    /// Transitions.
    pub transitions: Vec<StateTransition>,
    /// Maximum depth (number of transitions to reach accepting state).
    pub max_depth: usize,
}

impl InteractionStateMachine {
    /// Create a new FSM.
    pub fn new(name: impl Into<String>, element_id: ElementId) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            element_id,
            states: Vec::new(),
            transitions: Vec::new(),
            max_depth: crate::MAX_INTERACTION_DEPTH,
        }
    }

    /// Add a state.
    pub fn add_state(&mut self, state: InteractionState) {
        self.states.push(state);
    }

    /// Add a transition.
    pub fn add_transition(&mut self, transition: StateTransition) {
        self.transitions.push(transition);
    }

    /// Get the initial state(s).
    pub fn initial_states(&self) -> Vec<&InteractionState> {
        self.states.iter().filter(|s| s.is_initial).collect()
    }

    /// Get the accepting state(s).
    pub fn accepting_states(&self) -> Vec<&InteractionState> {
        self.states.iter().filter(|s| s.is_accepting).collect()
    }

    /// Get outgoing transitions from a given state.
    pub fn outgoing(&self, state_id: &str) -> Vec<&StateTransition> {
        self.transitions
            .iter()
            .filter(|t| t.from == state_id)
            .collect()
    }

    /// Get incoming transitions to a given state.
    pub fn incoming(&self, state_id: &str) -> Vec<&StateTransition> {
        self.transitions
            .iter()
            .filter(|t| t.to == state_id)
            .collect()
    }

    /// Find a state by ID.
    pub fn state(&self, id: &str) -> Option<&InteractionState> {
        self.states.iter().find(|s| s.id == id)
    }

    /// Validate the FSM structure.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Must have at least one initial state.
        if self.initial_states().is_empty() {
            errors.push("FSM has no initial state".into());
        }

        // Must have at least one accepting state.
        if self.accepting_states().is_empty() {
            errors.push("FSM has no accepting state".into());
        }

        // All transition endpoints must reference valid states.
        let state_ids: HashSet<&str> = self.states.iter().map(|s| s.id.as_str()).collect();
        for t in &self.transitions {
            if !state_ids.contains(t.from.as_str()) {
                errors.push(format!("Transition {} references unknown source state '{}'", t.id, t.from));
            }
            if !state_ids.contains(t.to.as_str()) {
                errors.push(format!("Transition {} references unknown target state '{}'", t.id, t.to));
            }
        }

        // State IDs must be unique.
        let mut seen = HashSet::new();
        for s in &self.states {
            if !seen.insert(&s.id) {
                errors.push(format!("Duplicate state ID: {}", s.id));
            }
        }

        // Transition IDs must be unique.
        let mut seen_t = HashSet::new();
        for t in &self.transitions {
            if !seen_t.insert(&t.id) {
                errors.push(format!("Duplicate transition ID: {}", t.id));
            }
        }

        errors
    }

    /// BFS reachability: return state IDs reachable from any initial state.
    pub fn reachable_states(&self) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        for s in self.initial_states() {
            queue.push_back(s.id.clone());
            visited.insert(s.id.clone());
        }

        while let Some(current) = queue.pop_front() {
            for t in self.outgoing(&current) {
                if visited.insert(t.to.clone()) {
                    queue.push_back(t.to.clone());
                }
            }
        }

        visited
    }

    /// Check if all accepting states are reachable from some initial state.
    pub fn all_accepting_reachable(&self) -> bool {
        let reachable = self.reachable_states();
        self.accepting_states()
            .iter()
            .all(|s| reachable.contains(&s.id))
    }

    /// Check for dead-end states (non-accepting states with no outgoing transitions).
    pub fn dead_end_states(&self) -> Vec<String> {
        self.states
            .iter()
            .filter(|s| !s.is_accepting && self.outgoing(&s.id).is_empty())
            .map(|s| s.id.clone())
            .collect()
    }

    /// Compute the minimum path length from initial to each accepting state.
    pub fn min_path_lengths(&self) -> HashMap<String, usize> {
        let mut distances: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        for s in self.initial_states() {
            queue.push_back((s.id.clone(), 0));
            distances.insert(s.id.clone(), 0);
        }

        while let Some((current, dist)) = queue.pop_front() {
            for t in self.outgoing(&current) {
                let new_dist = dist + 1;
                if !distances.contains_key(&t.to) || distances[&t.to] > new_dist {
                    distances.insert(t.to.clone(), new_dist);
                    queue.push_back((t.to.clone(), new_dist));
                }
            }
        }

        let accepting: HashSet<String> = self
            .accepting_states()
            .iter()
            .map(|s| s.id.clone())
            .collect();
        distances
            .into_iter()
            .filter(|(k, _)| accepting.contains(k))
            .collect()
    }

    /// Maximum depth of the FSM (longest shortest path to any accepting state).
    pub fn effective_depth(&self) -> usize {
        self.min_path_lengths()
            .values()
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Create a simple single-step interaction FSM.
    pub fn single_step(
        element_id: ElementId,
        interaction_type: InteractionType,
        guard: PoseGuard,
    ) -> Self {
        let mut fsm = Self::new(format!("single_{:?}", interaction_type), element_id);
        fsm.add_state(
            InteractionState::new("idle", "Idle")
                .initial()
                .with_interaction(interaction_type.clone()),
        );
        fsm.add_state(InteractionState::new("done", "Done").accepting());
        fsm.add_transition(
            StateTransition::new("t0", "idle", "done")
                .with_guard(guard)
                .with_trigger(interaction_type)
                .with_label("interact"),
        );
        fsm
    }

    /// Create a two-step approach-then-interact FSM.
    pub fn two_step(
        element_id: ElementId,
        approach_guard: PoseGuard,
        interact_guard: PoseGuard,
        interaction_type: InteractionType,
    ) -> Self {
        let mut fsm = Self::new(format!("two_step_{:?}", interaction_type), element_id);
        fsm.add_state(InteractionState::new("idle", "Idle").initial());
        fsm.add_state(
            InteractionState::new("approach", "Approach")
                .with_interaction(InteractionType::Proximity),
        );
        fsm.add_state(InteractionState::new("done", "Done").accepting());
        fsm.add_transition(
            StateTransition::new("t0", "idle", "approach")
                .with_guard(approach_guard)
                .with_label("approach"),
        );
        fsm.add_transition(
            StateTransition::new("t1", "approach", "done")
                .with_guard(interact_guard)
                .with_trigger(interaction_type)
                .with_label("interact"),
        );
        fsm
    }
}

impl std::fmt::Display for InteractionStateMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "FSM: {} ({})", self.name, self.id)?;
        writeln!(f, "  States:")?;
        for s in &self.states {
            writeln!(f, "    {s}")?;
        }
        writeln!(f, "  Transitions:")?;
        for t in &self.transitions {
            writeln!(f, "    {t}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InteractionSequence
// ---------------------------------------------------------------------------

/// A concrete sequence of transitions through an FSM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionSequence {
    /// FSM this sequence belongs to.
    pub fsm_id: Uuid,
    /// Ordered list of transition IDs taken.
    pub transitions: Vec<String>,
    /// Ordered list of state IDs visited.
    pub states: Vec<String>,
    /// Whether the sequence reaches an accepting state.
    pub accepted: bool,
    /// Body parameters used for this sequence.
    pub body_params: Option<Vec<f64>>,
    /// Joint angles at each step.
    pub joint_configs: Vec<Vec<f64>>,
}

impl InteractionSequence {
    /// Create a new empty sequence.
    pub fn new(fsm_id: Uuid) -> Self {
        Self {
            fsm_id,
            transitions: Vec::new(),
            states: Vec::new(),
            accepted: false,
            body_params: None,
            joint_configs: Vec::new(),
        }
    }

    /// Number of steps in the sequence.
    pub fn depth(&self) -> usize {
        self.transitions.len()
    }

    /// Whether the sequence is non-empty.
    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Record a step in the sequence.
    pub fn push_step(&mut self, state_id: String, transition_id: String, joints: Vec<f64>) {
        self.states.push(state_id);
        self.transitions.push(transition_id);
        self.joint_configs.push(joints);
    }

    /// Validate the sequence against a given FSM.
    pub fn validate_against(&self, fsm: &InteractionStateMachine) -> Vec<String> {
        let mut errors = Vec::new();

        // Check that states are valid.
        for sid in &self.states {
            if fsm.state(sid).is_none() {
                errors.push(format!("Unknown state in sequence: {sid}"));
            }
        }

        // Check transition connectivity.
        for i in 0..self.transitions.len() {
            let tid = &self.transitions[i];
            let transition = fsm.transitions.iter().find(|t| t.id == *tid);
            match transition {
                None => errors.push(format!("Unknown transition: {tid}")),
                Some(t) => {
                    if i > 0 && self.states[i - 1] != t.from {
                        errors.push(format!(
                            "Transition {} source '{}' does not match previous state '{}'",
                            tid, t.from, self.states[i - 1]
                        ));
                    }
                }
            }
        }

        errors
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_eid() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn make_simple_fsm() -> InteractionStateMachine {
        InteractionStateMachine::single_step(
            test_eid(),
            InteractionType::Click,
            PoseGuard::proximity([0.5, 1.0, 0.3], 0.1),
        )
    }

    fn make_two_step_fsm() -> InteractionStateMachine {
        InteractionStateMachine::two_step(
            test_eid(),
            PoseGuard::proximity([0.5, 1.0, 0.3], 0.5),
            PoseGuard::proximity([0.5, 1.0, 0.3], 0.05),
            InteractionType::Click,
        )
    }

    #[test]
    fn test_state_display() {
        let s = InteractionState::new("s0", "Start").initial();
        let display = format!("{s}");
        assert!(display.contains("s0"));
        assert!(display.contains("→"));
    }

    #[test]
    fn test_guard_trivial() {
        let g = PoseGuard::trivial();
        assert!(g.evaluate(&[0.0, 0.0, 0.0], &[]));
    }

    #[test]
    fn test_guard_proximity_pass() {
        let g = PoseGuard::proximity([0.0, 0.0, 0.0], 1.0);
        assert!(g.evaluate(&[0.5, 0.5, 0.5], &[]));
    }

    #[test]
    fn test_guard_proximity_fail() {
        let g = PoseGuard::proximity([0.0, 0.0, 0.0], 0.1);
        assert!(!g.evaluate(&[1.0, 1.0, 1.0], &[]));
    }

    #[test]
    fn test_guard_inside_box() {
        let g = PoseGuard::inside_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(g.evaluate(&[0.5, 0.5, 0.5], &[]));
        assert!(!g.evaluate(&[1.5, 0.5, 0.5], &[]));
    }

    #[test]
    fn test_guard_joint_range() {
        let g = PoseGuard::joint_range(0, -1.0, 1.0);
        assert!(g.evaluate(&[0.0, 0.0, 0.0], &[0.5, 0.0]));
        assert!(!g.evaluate(&[0.0, 0.0, 0.0], &[1.5, 0.0]));
    }

    #[test]
    fn test_guard_and() {
        let g = PoseGuard::and(vec![
            PoseGuard::proximity([0.0, 0.0, 0.0], 2.0),
            PoseGuard::inside_box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
        ]);
        assert!(g.evaluate(&[0.5, 0.5, 0.5], &[]));
        assert!(!g.evaluate(&[1.5, 0.5, 0.5], &[]));
    }

    #[test]
    fn test_guard_smt_proximity() {
        let g = PoseGuard::proximity([0.0, 0.0, 0.0], 1.0);
        let pos_vars = ["x".into(), "y".into(), "z".into()];
        let smt = g.to_smt(&pos_vars, &[]);
        let s = smt.to_smtlib();
        assert!(s.contains("<="));
        assert!(s.contains("x"));
    }

    #[test]
    fn test_guard_smt_box() {
        let g = PoseGuard::inside_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let pos_vars = ["x".into(), "y".into(), "z".into()];
        let smt = g.to_smt(&pos_vars, &[]);
        let s = smt.to_smtlib();
        assert!(s.contains("and"));
    }

    #[test]
    fn test_simple_fsm_validation() {
        let fsm = make_simple_fsm();
        let errors = fsm.validate();
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }

    #[test]
    fn test_simple_fsm_reachable() {
        let fsm = make_simple_fsm();
        let reachable = fsm.reachable_states();
        assert!(reachable.contains("idle"));
        assert!(reachable.contains("done"));
    }

    #[test]
    fn test_simple_fsm_accepting_reachable() {
        let fsm = make_simple_fsm();
        assert!(fsm.all_accepting_reachable());
    }

    #[test]
    fn test_simple_fsm_depth() {
        let fsm = make_simple_fsm();
        assert_eq!(fsm.effective_depth(), 1);
    }

    #[test]
    fn test_two_step_fsm() {
        let fsm = make_two_step_fsm();
        assert_eq!(fsm.states.len(), 3);
        assert_eq!(fsm.transitions.len(), 2);
        assert_eq!(fsm.effective_depth(), 2);
        assert!(fsm.validate().is_empty());
    }

    #[test]
    fn test_dead_end_detection() {
        let mut fsm = InteractionStateMachine::new("test", test_eid());
        fsm.add_state(InteractionState::new("s0", "Start").initial());
        fsm.add_state(InteractionState::new("s1", "Dead"));
        fsm.add_state(InteractionState::new("s2", "Done").accepting());
        fsm.add_transition(StateTransition::new("t0", "s0", "s1"));

        let dead = fsm.dead_end_states();
        assert_eq!(dead, vec!["s1"]);
    }

    #[test]
    fn test_fsm_display() {
        let fsm = make_simple_fsm();
        let display = format!("{fsm}");
        assert!(display.contains("FSM"));
        assert!(display.contains("States"));
        assert!(display.contains("Transitions"));
    }

    #[test]
    fn test_transition_display() {
        let t = StateTransition::new("t0", "idle", "done").with_label("click");
        let s = format!("{t}");
        assert!(s.contains("idle"));
        assert!(s.contains("click"));
        assert!(s.contains("done"));
    }

    #[test]
    fn test_interaction_sequence() {
        let fsm = make_simple_fsm();
        let mut seq = InteractionSequence::new(fsm.id);
        seq.push_step("idle".into(), "t0".into(), vec![0.0; 7]);
        seq.push_step("done".into(), "t0".into(), vec![0.0; 7]);
        assert_eq!(seq.depth(), 2);
        assert!(!seq.is_empty());
    }

    #[test]
    fn test_sequence_validate() {
        let fsm = make_simple_fsm();
        let mut seq = InteractionSequence::new(fsm.id);
        seq.states.push("idle".into());
        seq.transitions.push("t0".into());
        let errors = seq.validate_against(&fsm);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_sequence_validate_bad_state() {
        let fsm = make_simple_fsm();
        let mut seq = InteractionSequence::new(fsm.id);
        seq.states.push("nonexistent".into());
        seq.transitions.push("t0".into());
        let errors = seq.validate_against(&fsm);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_fsm_validate_duplicate_state() {
        let mut fsm = InteractionStateMachine::new("test", test_eid());
        fsm.add_state(InteractionState::new("s0", "A").initial());
        fsm.add_state(InteractionState::new("s0", "B").accepting());
        let errors = fsm.validate();
        assert!(errors.iter().any(|e| e.contains("Duplicate state")));
    }

    #[test]
    fn test_fsm_validate_missing_initial() {
        let mut fsm = InteractionStateMachine::new("test", test_eid());
        fsm.add_state(InteractionState::new("s0", "A").accepting());
        let errors = fsm.validate();
        assert!(errors.iter().any(|e| e.contains("no initial")));
    }

    #[test]
    fn test_min_path_lengths() {
        let fsm = make_two_step_fsm();
        let lengths = fsm.min_path_lengths();
        assert_eq!(lengths.get("done"), Some(&2));
    }

    #[test]
    fn test_outgoing_incoming() {
        let fsm = make_two_step_fsm();
        assert_eq!(fsm.outgoing("idle").len(), 1);
        assert_eq!(fsm.incoming("done").len(), 1);
        assert_eq!(fsm.incoming("idle").len(), 0);
    }

    #[test]
    fn test_fsm_serde_roundtrip() {
        let fsm = make_two_step_fsm();
        let json = serde_json::to_string(&fsm).unwrap();
        let back: InteractionStateMachine = serde_json::from_str(&json).unwrap();
        assert_eq!(fsm.states.len(), back.states.len());
        assert_eq!(fsm.transitions.len(), back.transitions.len());
    }
}

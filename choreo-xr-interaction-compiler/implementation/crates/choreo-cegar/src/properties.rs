//! Property specification: Safety, Liveness, Reachability, Deadlock-Freedom,
//! Determinism, Fairness. Also: monitors, negation, composition.

use std::collections::HashSet;
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{
    AutomatonDef, Guard, PredicateValuation, SpatialConstraint, SpatialPredicate,
    SpatialPredicateId, StateId, TransitionId,
};

// ---------------------------------------------------------------------------
// Property enum
// ---------------------------------------------------------------------------

/// A temporal / spatial property to verify.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Property {
    Safety(SafetyProperty),
    Liveness(LivenessProperty),
    Reachability(ReachabilityProperty),
    DeadlockFreedom,
    Determinism,
    Fairness(FairnessProperty),
}

impl Property {
    pub fn name(&self) -> &str {
        match self {
            Property::Safety(_) => "Safety",
            Property::Liveness(_) => "Liveness",
            Property::Reachability(_) => "Reachability",
            Property::DeadlockFreedom => "DeadlockFreedom",
            Property::Determinism => "Determinism",
            Property::Fairness(_) => "Fairness",
        }
    }

    pub fn is_safety(&self) -> bool {
        matches!(self, Property::Safety(_))
    }

    pub fn is_liveness(&self) -> bool {
        matches!(self, Property::Liveness(_))
    }

    pub fn is_reachability(&self) -> bool {
        matches!(self, Property::Reachability(_))
    }

    pub fn kind(&self) -> PropertyKind {
        match self {
            Property::Safety(_) => PropertyKind::Safety,
            Property::Liveness(_) => PropertyKind::Liveness,
            Property::Reachability(_) => PropertyKind::Reachability,
            Property::DeadlockFreedom => PropertyKind::Safety,
            Property::Determinism => PropertyKind::Safety,
            Property::Fairness(_) => PropertyKind::Liveness,
        }
    }

    /// Negate this property.
    pub fn negate(&self) -> Property {
        match self {
            Property::Safety(s) => Property::Reachability(ReachabilityProperty {
                target_predicate: s.bad_state_predicate.clone(),
            }),
            Property::Reachability(r) => Property::Safety(SafetyProperty {
                bad_state_predicate: r.target_predicate.clone(),
            }),
            Property::Liveness(l) => {
                // Negation of liveness "eventually progress" = safety "never progress"
                Property::Safety(SafetyProperty {
                    bad_state_predicate: l.progress_predicate.clone(),
                })
            }
            Property::DeadlockFreedom => {
                // Negation: "there exists a deadlock" = reachability of deadlock
                Property::Reachability(ReachabilityProperty {
                    target_predicate: SpatialConstraint::True,
                })
            }
            Property::Determinism => Property::Reachability(ReachabilityProperty {
                target_predicate: SpatialConstraint::True,
            }),
            Property::Fairness(f) => Property::Safety(SafetyProperty {
                bad_state_predicate: SpatialConstraint::Not(Box::new(f.fairness_predicate.clone())),
            }),
        }
    }

    /// Extract the spatial constraint, if any.
    pub fn constraint(&self) -> Option<&SpatialConstraint> {
        match self {
            Property::Safety(s) => Some(&s.bad_state_predicate),
            Property::Liveness(l) => Some(&l.progress_predicate),
            Property::Reachability(r) => Some(&r.target_predicate),
            Property::Fairness(f) => Some(&f.fairness_predicate),
            _ => None,
        }
    }
}

impl fmt::Display for Property {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Property::Safety(s) => write!(f, "Safety(AG ¬bad)"),
            Property::Liveness(l) => write!(f, "Liveness(GF progress)"),
            Property::Reachability(r) => write!(f, "Reachability(EF target)"),
            Property::DeadlockFreedom => write!(f, "DeadlockFreedom"),
            Property::Determinism => write!(f, "Determinism"),
            Property::Fairness(fa) => write!(f, "Fairness"),
        }
    }
}

// ---------------------------------------------------------------------------
// Property kinds
// ---------------------------------------------------------------------------

/// Classification of a property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyKind {
    Safety,
    Liveness,
    Reachability,
}

// ---------------------------------------------------------------------------
// Specific property types
// ---------------------------------------------------------------------------

/// A safety property: "no bad state is ever reached."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyProperty {
    /// The predicate characterising bad states.
    pub bad_state_predicate: SpatialConstraint,
}

impl SafetyProperty {
    pub fn new(bad_state_predicate: SpatialConstraint) -> Self {
        Self {
            bad_state_predicate,
        }
    }

    /// Check if a given predicate valuation is in a bad state.
    pub fn is_bad(&self, valuation: &PredicateValuation) -> bool {
        self.bad_state_predicate.evaluate(valuation).unwrap_or(false)
    }

    /// Get the set of predicate IDs referenced by this property.
    pub fn referenced_predicates(&self) -> HashSet<SpatialPredicateId> {
        self.bad_state_predicate.referenced_predicates().into_iter().collect()
    }
}

impl fmt::Display for SafetyProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AG ¬({:?})", self.bad_state_predicate)
    }
}

/// A liveness property: "progress is made infinitely often."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessProperty {
    /// The predicate characterising progress states.
    pub progress_predicate: SpatialConstraint,
}

impl LivenessProperty {
    pub fn new(progress_predicate: SpatialConstraint) -> Self {
        Self {
            progress_predicate,
        }
    }

    /// Check if a given valuation represents progress.
    pub fn is_progress(&self, valuation: &PredicateValuation) -> bool {
        self.progress_predicate.evaluate(valuation).unwrap_or(false)
    }

    pub fn referenced_predicates(&self) -> HashSet<SpatialPredicateId> {
        self.progress_predicate.referenced_predicates().into_iter().collect()
    }
}

impl fmt::Display for LivenessProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GF ({:?})", self.progress_predicate)
    }
}

/// A reachability property: "the target can be reached."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachabilityProperty {
    /// The predicate characterising the target states.
    pub target_predicate: SpatialConstraint,
}

impl ReachabilityProperty {
    pub fn new(target_predicate: SpatialConstraint) -> Self {
        Self { target_predicate }
    }

    /// Check if a given valuation is in a target state.
    pub fn is_target(&self, valuation: &PredicateValuation) -> bool {
        self.target_predicate.evaluate(valuation).unwrap_or(false)
    }

    pub fn referenced_predicates(&self) -> HashSet<SpatialPredicateId> {
        self.target_predicate.referenced_predicates().into_iter().collect()
    }
}

impl fmt::Display for ReachabilityProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EF ({:?})", self.target_predicate)
    }
}

/// A fairness property: "the fairness condition holds infinitely often."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessProperty {
    /// The fairness predicate.
    pub fairness_predicate: SpatialConstraint,
    /// Strong or weak fairness.
    pub kind: FairnessKind,
}

impl FairnessProperty {
    pub fn new(fairness_predicate: SpatialConstraint, kind: FairnessKind) -> Self {
        Self {
            fairness_predicate,
            kind,
        }
    }

    pub fn strong(predicate: SpatialConstraint) -> Self {
        Self::new(predicate, FairnessKind::Strong)
    }

    pub fn weak(predicate: SpatialConstraint) -> Self {
        Self::new(predicate, FairnessKind::Weak)
    }

    pub fn is_strong(&self) -> bool {
        self.kind == FairnessKind::Strong
    }

    pub fn referenced_predicates(&self) -> HashSet<SpatialPredicateId> {
        self.fairness_predicate.referenced_predicates().into_iter().collect()
    }
}

impl fmt::Display for FairnessProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            FairnessKind::Strong => "strong",
            FairnessKind::Weak => "weak",
        };
        write!(f, "Fairness({}, {:?})", kind, self.fairness_predicate)
    }
}

/// Strong vs. weak fairness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FairnessKind {
    Strong,
    Weak,
}

// ---------------------------------------------------------------------------
// Property combinators
// ---------------------------------------------------------------------------

/// Conjunction of multiple properties.
#[derive(Debug, Clone)]
pub struct PropertyConjunction {
    pub properties: Vec<Property>,
}

impl PropertyConjunction {
    pub fn new(properties: Vec<Property>) -> Self {
        Self { properties }
    }

    pub fn add(&mut self, property: Property) {
        self.properties.push(property);
    }

    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    pub fn len(&self) -> usize {
        self.properties.len()
    }
}

/// Disjunction of multiple properties.
#[derive(Debug, Clone)]
pub struct PropertyDisjunction {
    pub properties: Vec<Property>,
}

impl PropertyDisjunction {
    pub fn new(properties: Vec<Property>) -> Self {
        Self { properties }
    }

    pub fn add(&mut self, property: Property) {
        self.properties.push(property);
    }
}

// ---------------------------------------------------------------------------
// Monitor automaton
// ---------------------------------------------------------------------------

/// A monitor automaton that tracks property satisfaction at runtime.
#[derive(Debug, Clone)]
pub struct PropertyMonitor {
    /// The property being monitored.
    pub property: Property,
    /// Current monitor state.
    pub state: MonitorState,
    /// Number of steps observed.
    pub steps: usize,
    /// Number of times the property was satisfied.
    pub satisfaction_count: usize,
    /// Number of times the property was violated.
    pub violation_count: usize,
    /// History of monitor states.
    pub history: Vec<MonitorState>,
    /// Maximum history to keep.
    pub max_history: usize,
}

/// State of a property monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonitorState {
    /// Property holds so far.
    Accepting,
    /// Property is violated.
    Rejecting,
    /// Cannot yet determine.
    Pending,
}

impl PropertyMonitor {
    pub fn new(property: Property) -> Self {
        Self {
            property,
            state: MonitorState::Pending,
            steps: 0,
            satisfaction_count: 0,
            violation_count: 0,
            history: Vec::new(),
            max_history: 10000,
        }
    }

    /// Feed one observation to the monitor.
    pub fn observe(&mut self, valuation: &PredicateValuation) {
        self.steps += 1;

        let new_state = match &self.property {
            Property::Safety(s) => {
                if s.is_bad(valuation) {
                    MonitorState::Rejecting
                } else {
                    MonitorState::Accepting
                }
            }
            Property::Liveness(l) => {
                if l.is_progress(valuation) {
                    self.satisfaction_count += 1;
                    MonitorState::Accepting
                } else {
                    MonitorState::Pending
                }
            }
            Property::Reachability(r) => {
                if r.is_target(valuation) {
                    MonitorState::Accepting
                } else {
                    MonitorState::Pending
                }
            }
            Property::DeadlockFreedom => {
                // Deadlock detection requires checking outgoing transitions,
                // not just valuations. Mark as pending.
                MonitorState::Pending
            }
            Property::Determinism => MonitorState::Pending,
            Property::Fairness(f) => {
                if f.fairness_predicate.evaluate(valuation).unwrap_or(false) {
                    self.satisfaction_count += 1;
                    MonitorState::Accepting
                } else {
                    MonitorState::Pending
                }
            }
        };

        // Update state: once rejecting, stay rejecting for safety
        match &self.property {
            Property::Safety(_) => {
                if new_state == MonitorState::Rejecting {
                    self.violation_count += 1;
                    self.state = MonitorState::Rejecting;
                } else if self.state != MonitorState::Rejecting {
                    self.state = new_state;
                }
            }
            Property::Reachability(_) => {
                if new_state == MonitorState::Accepting {
                    self.state = MonitorState::Accepting;
                }
            }
            _ => {
                self.state = new_state;
            }
        }

        if self.history.len() < self.max_history {
            self.history.push(new_state);
        }
    }

    /// Reset the monitor to its initial state.
    pub fn reset(&mut self) {
        self.state = MonitorState::Pending;
        self.steps = 0;
        self.satisfaction_count = 0;
        self.violation_count = 0;
        self.history.clear();
    }

    /// Get the current verdict.
    pub fn verdict(&self) -> MonitorVerdict {
        match self.state {
            MonitorState::Accepting => MonitorVerdict::Satisfied,
            MonitorState::Rejecting => MonitorVerdict::Violated,
            MonitorState::Pending => MonitorVerdict::Inconclusive,
        }
    }

    pub fn is_accepting(&self) -> bool {
        self.state == MonitorState::Accepting
    }

    pub fn is_rejecting(&self) -> bool {
        self.state == MonitorState::Rejecting
    }
}

/// Verdict of a property monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonitorVerdict {
    Satisfied,
    Violated,
    Inconclusive,
}

// ---------------------------------------------------------------------------
// Property parser (simple syntax)
// ---------------------------------------------------------------------------

/// Parse a property from a simple string specification.
///
/// Supported syntax:
/// - `safety: pred_name`  — safety property "never pred_name"
/// - `liveness: pred_name` — liveness "infinitely often pred_name"
/// - `reachability: pred_name` — reachability "eventually pred_name"
/// - `deadlock-freedom` — deadlock freedom
/// - `determinism` — determinism
/// - `fairness: pred_name` — weak fairness on pred_name
/// - `strong-fairness: pred_name` — strong fairness on pred_name
pub fn parse_property(spec: &str) -> Result<Property, String> {
    let spec = spec.trim();

    if spec.eq_ignore_ascii_case("deadlock-freedom") || spec.eq_ignore_ascii_case("deadlockfreedom")
    {
        return Ok(Property::DeadlockFreedom);
    }
    if spec.eq_ignore_ascii_case("determinism") {
        return Ok(Property::Determinism);
    }

    let (kind, rest) = spec
        .split_once(':')
        .ok_or_else(|| format!("Invalid property spec: {}", spec))?;
    let predicate_name = rest.trim();

    if predicate_name.is_empty() {
        return Err("Empty predicate name".to_string());
    }

    let pred_id = SpatialPredicateId(hash_string(predicate_name));
    let constraint = SpatialConstraint::Predicate(pred_id);

    match kind.trim().to_lowercase().as_str() {
        "safety" => Ok(Property::Safety(SafetyProperty::new(constraint))),
        "liveness" => Ok(Property::Liveness(LivenessProperty::new(constraint))),
        "reachability" => Ok(Property::Reachability(ReachabilityProperty::new(constraint))),
        "fairness" => Ok(Property::Fairness(FairnessProperty::weak(constraint))),
        "strong-fairness" => Ok(Property::Fairness(FairnessProperty::strong(constraint))),
        _ => Err(format!("Unknown property kind: {}", kind.trim())),
    }
}

/// Simple string hash for predicate name → id mapping.
fn hash_string(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

// ---------------------------------------------------------------------------
// Property set
// ---------------------------------------------------------------------------

/// A named set of properties to verify together.
#[derive(Debug, Clone)]
pub struct PropertySet {
    pub name: String,
    pub properties: Vec<(String, Property)>,
}

impl PropertySet {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            properties: Vec::new(),
        }
    }

    pub fn add(&mut self, name: impl Into<String>, property: Property) {
        self.properties.push((name.into(), property));
    }

    pub fn len(&self) -> usize {
        self.properties.len()
    }

    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(String, Property)> {
        self.properties.iter()
    }

    /// Get all safety properties.
    pub fn safety_properties(&self) -> Vec<&SafetyProperty> {
        self.properties
            .iter()
            .filter_map(|(_, p)| match p {
                Property::Safety(s) => Some(s),
                _ => None,
            })
            .collect()
    }

    /// Get all liveness properties.
    pub fn liveness_properties(&self) -> Vec<&LivenessProperty> {
        self.properties
            .iter()
            .filter_map(|(_, p)| match p {
                Property::Liveness(l) => Some(l),
                _ => None,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Automaton product for property checking
// ---------------------------------------------------------------------------

/// Build a product automaton of the system automaton with a property monitor.
///
/// This creates a combined automaton where accepting states represent violation
/// of the property (for safety) or satisfaction (for reachability).
pub fn build_product_automaton(
    system: &AutomatonDef,
    property: &Property,
) -> ProductAutomaton {
    let mut product = ProductAutomaton {
        states: Vec::new(),
        transitions: Vec::new(),
        initial: Vec::new(),
        accepting: Vec::new(),
        property_kind: property.kind(),
    };

    // For safety: accepting = system_state × bad_predicate_holds
    // For reachability: accepting = system_state × target_predicate_holds
    // We encode monitor states as offsets

    let n = system.states.len();

    // Create product states: each (system_state, monitor_state)
    // monitor_state ∈ {0 = accepting, 1 = rejecting}
    for (i, state) in system.states.iter().enumerate() {
        let prod_accepting = ProductState {
            system_state: state.id,
            monitor_state: 0,
            label: format!("({}, accept)", state.name),
        };
        let prod_rejecting = ProductState {
            system_state: state.id,
            monitor_state: 1,
            label: format!("({}, reject)", state.name),
        };
        product.states.push(prod_accepting);
        product.states.push(prod_rejecting);
    }

    // Initial states: system initial × monitor accepting
    product.initial.push(ProductStateId {
        system: system.initial,
        monitor: 0,
    });

    // Transitions: mirror system transitions with monitor updates
    for t in &system.transitions {
        // Transition in accepting monitor state
        product.transitions.push(ProductTransition {
            source: ProductStateId {
                system: t.source,
                monitor: 0,
            },
            target: ProductStateId {
                system: t.target,
                monitor: 0,
            },
            guard: t.guard.clone(),
            original: t.id,
        });
        // Transition in rejecting monitor state
        product.transitions.push(ProductTransition {
            source: ProductStateId {
                system: t.source,
                monitor: 1,
            },
            target: ProductStateId {
                system: t.target,
                monitor: 1,
            },
            guard: t.guard.clone(),
            original: t.id,
        });
    }

    // Accepting states depend on property type
    match property {
        Property::Safety(_) => {
            // Rejecting monitor states are accepting (we look for counterexamples)
            for state in &system.states {
                product.accepting.push(ProductStateId {
                    system: state.id,
                    monitor: 1,
                });
            }
        }
        Property::Reachability(_) => {
            // Target states in accepting monitor
            for state in &system.states {
                product.accepting.push(ProductStateId {
                    system: state.id,
                    monitor: 0,
                });
            }
        }
        _ => {}
    }

    product
}

/// A state in the product automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductState {
    pub system_state: StateId,
    pub monitor_state: u32,
    pub label: String,
}

/// ID for a product state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductStateId {
    pub system: StateId,
    pub monitor: u32,
}

/// A transition in the product automaton.
#[derive(Debug, Clone)]
pub struct ProductTransition {
    pub source: ProductStateId,
    pub target: ProductStateId,
    pub guard: Guard,
    pub original: TransitionId,
}

/// The product automaton.
#[derive(Debug, Clone)]
pub struct ProductAutomaton {
    pub states: Vec<ProductState>,
    pub transitions: Vec<ProductTransition>,
    pub initial: Vec<ProductStateId>,
    pub accepting: Vec<ProductStateId>,
    pub property_kind: PropertyKind,
}

impl ProductAutomaton {
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_accepting(&self, id: &ProductStateId) -> bool {
        self.accepting.contains(id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, State, Transition};

    #[test]
    fn test_property_kind() {
        let safety = Property::Safety(SafetyProperty::new(SpatialConstraint::True));
        assert_eq!(safety.kind(), PropertyKind::Safety);
        assert!(safety.is_safety());
        assert!(!safety.is_liveness());
        assert_eq!(safety.name(), "Safety");
    }

    #[test]
    fn test_liveness_property() {
        let live = Property::Liveness(LivenessProperty::new(SpatialConstraint::True));
        assert_eq!(live.kind(), PropertyKind::Liveness);
        assert!(live.is_liveness());
    }

    #[test]
    fn test_reachability_property() {
        let reach = Property::Reachability(ReachabilityProperty::new(SpatialConstraint::True));
        assert_eq!(reach.kind(), PropertyKind::Reachability);
        assert!(reach.is_reachability());
    }

    #[test]
    fn test_property_negation() {
        let safety = Property::Safety(SafetyProperty::new(SpatialConstraint::True));
        let neg = safety.negate();
        assert!(neg.is_reachability());

        let reach = Property::Reachability(ReachabilityProperty::new(SpatialConstraint::True));
        let neg = reach.negate();
        assert!(neg.is_safety());
    }

    #[test]
    fn test_deadlock_freedom() {
        let prop = Property::DeadlockFreedom;
        assert_eq!(prop.kind(), PropertyKind::Safety);
        assert_eq!(prop.name(), "DeadlockFreedom");
    }

    #[test]
    fn test_determinism() {
        let prop = Property::Determinism;
        assert_eq!(prop.name(), "Determinism");
    }

    #[test]
    fn test_fairness_property() {
        let strong =
            FairnessProperty::strong(SpatialConstraint::True);
        assert!(strong.is_strong());

        let weak = FairnessProperty::weak(SpatialConstraint::True);
        assert!(!weak.is_strong());
    }

    #[test]
    fn test_safety_is_bad() {
        let safety = SafetyProperty::new(SpatialConstraint::True);
        let valuation = PredicateValuation::new();
        assert!(safety.is_bad(&valuation)); // True always holds
    }

    #[test]
    fn test_parse_property_safety() {
        let prop = parse_property("safety: collision").unwrap();
        assert!(prop.is_safety());
    }

    #[test]
    fn test_parse_property_liveness() {
        let prop = parse_property("liveness: progress").unwrap();
        assert!(prop.is_liveness());
    }

    #[test]
    fn test_parse_property_deadlock() {
        let prop = parse_property("deadlock-freedom").unwrap();
        assert_eq!(prop.name(), "DeadlockFreedom");
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_property("").is_err());
        assert!(parse_property("unknown: x").is_err());
    }

    #[test]
    fn test_property_display() {
        let safety = Property::Safety(SafetyProperty::new(SpatialConstraint::True));
        let s = format!("{}", safety);
        assert!(s.contains("Safety"));
    }

    #[test]
    fn test_property_conjunction() {
        let mut conj = PropertyConjunction::new(vec![]);
        assert!(conj.is_empty());
        conj.add(Property::DeadlockFreedom);
        conj.add(Property::Determinism);
        assert_eq!(conj.len(), 2);
    }

    #[test]
    fn test_property_monitor_safety() {
        let safety = Property::Safety(SafetyProperty::new(
            SpatialConstraint::Predicate(SpatialPredicateId(42), true),
        ));
        let mut monitor = PropertyMonitor::new(safety);

        // Observe a good state (predicate 42 is false → not bad)
        let val = PredicateValuation::new();
        monitor.observe(&val);
        assert!(monitor.is_accepting());

        // Observe a bad state (predicate 42 is true → bad)
        let mut bad_val = PredicateValuation::new();
        bad_val.set(SpatialPredicateId(42), true);
        monitor.observe(&bad_val);
        assert!(monitor.is_rejecting());
        assert_eq!(monitor.verdict(), MonitorVerdict::Violated);
    }

    #[test]
    fn test_property_monitor_reset() {
        let safety = Property::Safety(SafetyProperty::new(SpatialConstraint::True));
        let mut monitor = PropertyMonitor::new(safety);
        monitor.observe(&PredicateValuation::new());
        monitor.reset();
        assert_eq!(monitor.steps, 0);
        assert_eq!(monitor.state, MonitorState::Pending);
    }

    #[test]
    fn test_property_set() {
        let mut set = PropertySet::new("test_props");
        set.add("no_collision", Property::Safety(SafetyProperty::new(SpatialConstraint::True)));
        set.add("deadlock_free", Property::DeadlockFreedom);
        assert_eq!(set.len(), 2);
        assert_eq!(set.safety_properties().len(), 1);
        assert_eq!(set.liveness_properties().len(), 0);
    }

    #[test]
    fn test_product_automaton() {
        let automaton = AutomatonDef {
            states: vec![
                State {
                    id: StateId(0),
                    name: "s0".into(),
                    invariant: None,
                    is_accepting: false,
                },
                State {
                    id: StateId(1),
                    name: "s1".into(),
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
        };

        let property = Property::Safety(SafetyProperty::new(SpatialConstraint::True));
        let product = build_product_automaton(&automaton, &property);

        assert_eq!(product.state_count(), 4); // 2 system × 2 monitor
        assert_eq!(product.transition_count(), 2); // 1 system × 2 monitor states
        assert!(!product.initial.is_empty());
    }

    #[test]
    fn test_property_constraint_extraction() {
        let safety = Property::Safety(SafetyProperty::new(SpatialConstraint::True));
        assert!(safety.constraint().is_some());

        let dl = Property::DeadlockFreedom;
        assert!(dl.constraint().is_none());
    }
}

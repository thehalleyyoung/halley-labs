//! EC domain definitions for XR interactions.
//!
//! Pre-built axiom sets for common XR interaction patterns such as
//! grab/release, gaze-dwell, proximity triggers, hand menus, and
//! two-hand manipulation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::axioms::*;
use crate::fluent::*;
use crate::local_types::*;

// ─── XRDomain ────────────────────────────────────────────────────────────────

/// A complete XR interaction domain with axioms, fluents, and predicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XRDomain {
    pub name: String,
    pub axiom_set: AxiomSet,
    pub fluent_store: FluentStore,
    pub spatial_predicates: Vec<(SpatialPredicateId, SpatialPredicate)>,
    pub description: String,
}

impl XRDomain {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            axiom_set: AxiomSet::new(),
            fluent_store: FluentStore::new(),
            spatial_predicates: Vec::new(),
            description: String::new(),
        }
    }

    /// Pre-built domain for grab/release interaction.
    pub fn grab_release() -> Self {
        let mut domain = Self::new("grab_release");
        domain.description = "Standard grab and release interaction with near predicate".into();

        // Fluents
        let near_id = domain.fluent_store.insert(Fluent::boolean("near_hand_object", false));
        let grabbed_id = domain.fluent_store.insert(Fluent::boolean("grabbed", false));
        let held_id = domain.fluent_store.insert(Fluent::boolean("held", false));

        // Spatial predicate: hand near object
        let sp_near = SpatialPredicateId(1);
        domain.spatial_predicates.push((
            sp_near,
            SpatialPredicate::Near {
                a: EntityId(0),
                b: EntityId(0),
                distance: 0.15,
            },
        ));

        let mut next_id = 1u64;

        // Axiom: grab gesture + near → initiated grabbed
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "grab_initiate".into(),
            conditions: vec![AxiomCondition::FluentHolds(near_id)],
            fluent: grabbed_id,
            event: EventPattern::GestureMatch(GestureType::Grab),
            new_value: Fluent::boolean("grabbed", true),
            priority: 0,
        });
        next_id += 1;

        // Axiom: release action → terminate grabbed
        domain.axiom_set.add(Axiom::TerminationAxiom {
            id: AxiomId(next_id),
            name: "grab_terminate".into(),
            conditions: vec![],
            fluent: grabbed_id,
            event: EventPattern::ActionMatch(ActionType::Deactivate),
            priority: 0,
        });
        next_id += 1;

        // Axiom: grab → start holding (causal)
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "hold_initiate".into(),
            conditions: vec![AxiomCondition::FluentHolds(grabbed_id)],
            fluent: held_id,
            event: EventPattern::Any,
            new_value: Fluent::boolean("held", true),
            priority: -1,
        });
        next_id += 1;

        // Axiom: release → stop holding
        domain.axiom_set.add(Axiom::TerminationAxiom {
            id: AxiomId(next_id),
            name: "hold_terminate".into(),
            conditions: vec![],
            fluent: held_id,
            event: EventPattern::ActionMatch(ActionType::Deactivate),
            priority: -1,
        });

        domain
    }

    /// Pre-built domain for gaze-dwell activation.
    pub fn gaze_dwell() -> Self {
        let mut domain = Self::new("gaze_dwell");
        domain.description = "Gaze-dwell activation: look at target for N seconds to activate".into();

        let gazing_id = domain.fluent_store.insert(Fluent::boolean("gazing_at_target", false));
        let _dwell_timer_id = domain.fluent_store.insert(
            Fluent::timer("dwell_timer", Duration::from_secs(0.0)),
        );
        let activated_id = domain.fluent_store.insert(Fluent::boolean("activated", false));

        let mut next_id = 1u64;

        // Gaze enter → start gazing
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "gaze_enter".into(),
            conditions: vec![],
            fluent: gazing_id,
            event: EventPattern::KindMatch(EventKind::GazeEnter { entity: EntityId(0) }),
            new_value: Fluent::boolean("gazing_at_target", true),
            priority: 0,
        });
        next_id += 1;

        // Gaze exit → stop gazing
        domain.axiom_set.add(Axiom::TerminationAxiom {
            id: AxiomId(next_id),
            name: "gaze_exit".into(),
            conditions: vec![],
            fluent: gazing_id,
            event: EventPattern::KindMatch(EventKind::GazeExit { entity: EntityId(0) }),
            priority: 0,
        });
        next_id += 1;

        // Gaze enter causes timer start
        domain.axiom_set.add(Axiom::CausalAxiom {
            id: AxiomId(next_id),
            name: "gaze_start_timer".into(),
            cause_event: EventPattern::KindMatch(EventKind::GazeEnter { entity: EntityId(0) }),
            conditions: vec![],
            effect_event: EventKind::TimerStarted {
                name: "dwell_timer".into(),
                duration: Duration::from_secs(2.0),
            },
            delay: None,
        });
        next_id += 1;

        // Timer expired + still gazing → activate
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "dwell_activate".into(),
            conditions: vec![AxiomCondition::FluentHolds(gazing_id)],
            fluent: activated_id,
            event: EventPattern::KindMatch(EventKind::TimerExpired {
                name: "dwell_timer".into(),
            }),
            new_value: Fluent::boolean("activated", true),
            priority: 0,
        });

        domain
    }

    /// Pre-built domain for proximity-triggered events.
    pub fn proximity_trigger() -> Self {
        let mut domain = Self::new("proximity_trigger");
        domain.description = "Proximity-triggered activation when user enters a zone".into();

        let in_zone_id = domain.fluent_store.insert(Fluent::boolean("in_zone", false));
        let triggered_id = domain.fluent_store.insert(Fluent::boolean("triggered", false));
        let cooldown_id = domain.fluent_store.insert(Fluent::boolean("cooldown", false));

        let sp_inside = SpatialPredicateId(1);
        domain.spatial_predicates.push((
            sp_inside,
            SpatialPredicate::Inside {
                entity: EntityId(0),
                region: RegionId(1),
            },
        ));

        let mut next_id = 1u64;

        // Spatial change (entered zone) → in_zone
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "enter_zone".into(),
            conditions: vec![AxiomCondition::FluentNotHolds(cooldown_id)],
            fluent: in_zone_id,
            event: EventPattern::SpatialChangeMatch(sp_inside),
            new_value: Fluent::boolean("in_zone", true),
            priority: 0,
        });
        next_id += 1;

        // In zone → trigger
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "proximity_trigger".into(),
            conditions: vec![
                AxiomCondition::FluentHolds(in_zone_id),
                AxiomCondition::FluentNotHolds(cooldown_id),
            ],
            fluent: triggered_id,
            event: EventPattern::Any,
            new_value: Fluent::boolean("triggered", true),
            priority: 0,
        });
        next_id += 1;

        // Trigger → start cooldown (causal)
        domain.axiom_set.add(Axiom::CausalAxiom {
            id: AxiomId(next_id),
            name: "start_cooldown".into(),
            cause_event: EventPattern::SpatialChangeMatch(sp_inside),
            conditions: vec![AxiomCondition::FluentHolds(triggered_id)],
            effect_event: EventKind::TimerStarted {
                name: "cooldown_timer".into(),
                duration: Duration::from_secs(5.0),
            },
            delay: None,
        });
        next_id += 1;

        // Cooldown timer expired → terminate cooldown
        domain.axiom_set.add(Axiom::TerminationAxiom {
            id: AxiomId(next_id),
            name: "cooldown_end".into(),
            conditions: vec![],
            fluent: cooldown_id,
            event: EventPattern::KindMatch(EventKind::TimerExpired {
                name: "cooldown_timer".into(),
            }),
            priority: 0,
        });

        domain
    }

    /// Pre-built domain for hand menu interaction.
    pub fn hand_menu() -> Self {
        let mut domain = Self::new("hand_menu");
        domain.description = "Palm-up hand menu with gaze selection".into();

        let _palm_up_id = domain.fluent_store.insert(Fluent::boolean("palm_up", false));
        let menu_visible_id = domain.fluent_store.insert(Fluent::boolean("menu_visible", false));
        let item_hovered_id = domain.fluent_store.insert(Fluent::boolean("item_hovered", false));
        let item_selected_id = domain.fluent_store.insert(Fluent::boolean("item_selected", false));

        let mut next_id = 1u64;

        // Palm facing up → show menu
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "palm_up_show_menu".into(),
            conditions: vec![],
            fluent: menu_visible_id,
            event: EventPattern::NamedMatch("palm_up".into()),
            new_value: Fluent::boolean("menu_visible", true),
            priority: 0,
        });
        next_id += 1;

        // Palm down → hide menu
        domain.axiom_set.add(Axiom::TerminationAxiom {
            id: AxiomId(next_id),
            name: "palm_down_hide_menu".into(),
            conditions: vec![],
            fluent: menu_visible_id,
            event: EventPattern::NamedMatch("palm_down".into()),
            priority: 0,
        });
        next_id += 1;

        // Gaze enters menu item while menu visible → hover
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "menu_item_hover".into(),
            conditions: vec![AxiomCondition::FluentHolds(menu_visible_id)],
            fluent: item_hovered_id,
            event: EventPattern::KindMatch(EventKind::GazeEnter { entity: EntityId(0) }),
            new_value: Fluent::boolean("item_hovered", true),
            priority: 0,
        });
        next_id += 1;

        // Gaze exits menu item → unhover
        domain.axiom_set.add(Axiom::TerminationAxiom {
            id: AxiomId(next_id),
            name: "menu_item_unhover".into(),
            conditions: vec![],
            fluent: item_hovered_id,
            event: EventPattern::KindMatch(EventKind::GazeExit { entity: EntityId(0) }),
            priority: 0,
        });
        next_id += 1;

        // Pinch while hovering → select
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "menu_item_select".into(),
            conditions: vec![
                AxiomCondition::FluentHolds(menu_visible_id),
                AxiomCondition::FluentHolds(item_hovered_id),
            ],
            fluent: item_selected_id,
            event: EventPattern::GestureMatch(GestureType::Pinch),
            new_value: Fluent::boolean("item_selected", true),
            priority: 0,
        });

        domain
    }

    /// Pre-built domain for two-hand manipulation.
    pub fn two_hand_manipulation() -> Self {
        let mut domain = Self::new("two_hand_manipulation");
        domain.description = "Two-hand scale/rotate manipulation".into();

        let left_grab_id = domain.fluent_store.insert(Fluent::boolean("left_grab", false));
        let right_grab_id = domain.fluent_store.insert(Fluent::boolean("right_grab", false));
        let bimanual_id = domain.fluent_store.insert(Fluent::boolean("bimanual_active", false));
        let scaling_id = domain.fluent_store.insert(Fluent::boolean("scaling", false));
        let rotating_id = domain.fluent_store.insert(Fluent::boolean("rotating", false));

        let sp_near_left = SpatialPredicateId(1);
        let sp_near_right = SpatialPredicateId(2);

        domain.spatial_predicates.push((
            sp_near_left,
            SpatialPredicate::Near { a: EntityId(1), b: EntityId(0), distance: 0.15 },
        ));
        domain.spatial_predicates.push((
            sp_near_right,
            SpatialPredicate::Near { a: EntityId(2), b: EntityId(0), distance: 0.15 },
        ));

        let mut next_id = 1u64;

        // Left grab
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "left_grab".into(),
            conditions: vec![],
            fluent: left_grab_id,
            event: EventPattern::KindMatch(EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Left,
                entity: EntityId(0),
            }),
            new_value: Fluent::boolean("left_grab", true),
            priority: 0,
        });
        next_id += 1;

        // Right grab
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "right_grab".into(),
            conditions: vec![],
            fluent: right_grab_id,
            event: EventPattern::KindMatch(EventKind::Gesture {
                gesture: GestureType::Grab,
                hand: HandSide::Right,
                entity: EntityId(0),
            }),
            new_value: Fluent::boolean("right_grab", true),
            priority: 0,
        });
        next_id += 1;

        // Both hands grabbing → bimanual active
        domain.axiom_set.add(Axiom::StateConstraint {
            id: AxiomId(next_id),
            name: "bimanual_constraint".into(),
            conditions: vec![
                AxiomCondition::FluentHolds(left_grab_id),
                AxiomCondition::FluentHolds(right_grab_id),
            ],
            fluent: bimanual_id,
            required_value: Fluent::boolean("bimanual_active", true),
        });
        next_id += 1;

        // Bimanual + distance increasing → scaling
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "bimanual_scale".into(),
            conditions: vec![AxiomCondition::FluentHolds(bimanual_id)],
            fluent: scaling_id,
            event: EventPattern::NamedMatch("distance_change".into()),
            new_value: Fluent::boolean("scaling", true),
            priority: 0,
        });
        next_id += 1;

        // Bimanual + rotation detected → rotating
        domain.axiom_set.add(Axiom::InitiationAxiom {
            id: AxiomId(next_id),
            name: "bimanual_rotate".into(),
            conditions: vec![AxiomCondition::FluentHolds(bimanual_id)],
            fluent: rotating_id,
            event: EventPattern::NamedMatch("rotation_change".into()),
            new_value: Fluent::boolean("rotating", true),
            priority: 0,
        });

        domain
    }

    /// Get the number of axioms in this domain.
    pub fn axiom_count(&self) -> usize {
        self.axiom_set.len()
    }

    /// Get the number of fluents in this domain.
    pub fn fluent_count(&self) -> usize {
        self.fluent_store.len()
    }
}

// ─── DomainBuilder ───────────────────────────────────────────────────────────

/// Builder for constructing custom XR interaction domains.
pub struct DomainBuilder {
    domain: XRDomain,
    next_axiom_id: u64,
}

impl DomainBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            domain: XRDomain::new(name),
            next_axiom_id: 1,
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.domain.description = desc.into();
        self
    }

    pub fn add_boolean_fluent(mut self, name: impl Into<String>, initial: bool) -> (Self, FluentId) {
        let id = self.domain.fluent_store.insert(Fluent::boolean(name, initial));
        (self, id)
    }

    pub fn add_numeric_fluent(mut self, name: impl Into<String>, initial: f64) -> (Self, FluentId) {
        let id = self.domain.fluent_store.insert(Fluent::numeric(name, initial));
        (self, id)
    }

    pub fn add_timer_fluent(mut self, name: impl Into<String>) -> (Self, FluentId) {
        let id = self.domain.fluent_store.insert(
            Fluent::timer(name, Duration::from_secs(0.0)),
        );
        (self, id)
    }

    pub fn add_spatial_predicate(
        mut self,
        predicate: SpatialPredicate,
    ) -> (Self, SpatialPredicateId) {
        let id = SpatialPredicateId(self.domain.spatial_predicates.len() as u64 + 1);
        self.domain.spatial_predicates.push((id, predicate));
        (self, id)
    }

    pub fn add_initiation(
        mut self,
        name: impl Into<String>,
        event: EventPattern,
        fluent: FluentId,
        new_value: Fluent,
        conditions: Vec<AxiomCondition>,
    ) -> Self {
        let id = AxiomId(self.next_axiom_id);
        self.next_axiom_id += 1;
        self.domain.axiom_set.add(Axiom::InitiationAxiom {
            id,
            name: name.into(),
            conditions,
            fluent,
            event,
            new_value,
            priority: 0,
        });
        self
    }

    pub fn add_termination(
        mut self,
        name: impl Into<String>,
        event: EventPattern,
        fluent: FluentId,
        conditions: Vec<AxiomCondition>,
    ) -> Self {
        let id = AxiomId(self.next_axiom_id);
        self.next_axiom_id += 1;
        self.domain.axiom_set.add(Axiom::TerminationAxiom {
            id,
            name: name.into(),
            conditions,
            fluent,
            event,
            priority: 0,
        });
        self
    }

    pub fn add_state_constraint(
        mut self,
        name: impl Into<String>,
        fluent: FluentId,
        required_value: Fluent,
        conditions: Vec<AxiomCondition>,
    ) -> Self {
        let id = AxiomId(self.next_axiom_id);
        self.next_axiom_id += 1;
        self.domain.axiom_set.add(Axiom::StateConstraint {
            id,
            name: name.into(),
            conditions,
            fluent,
            required_value,
        });
        self
    }

    pub fn add_causal(
        mut self,
        name: impl Into<String>,
        cause: EventPattern,
        effect: EventKind,
        conditions: Vec<AxiomCondition>,
    ) -> Self {
        let id = AxiomId(self.next_axiom_id);
        self.next_axiom_id += 1;
        self.domain.axiom_set.add(Axiom::CausalAxiom {
            id,
            name: name.into(),
            cause_event: cause,
            conditions,
            effect_event: effect,
            delay: None,
        });
        self
    }

    pub fn build(self) -> ECResult<XRDomain> {
        let warnings = validate_domain(&self.domain);
        if warnings.iter().any(|w| w.contains("circular")) {
            return Err(ECError::CircularDependency(
                warnings.join("; "),
            ));
        }
        Ok(self.domain)
    }
}

// ─── Domain validation ───────────────────────────────────────────────────────

/// Validate a domain for circular dependencies and contradictions.
pub fn validate_domain(domain: &XRDomain) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check for circular initiation/termination
    for (_, axiom) in domain.axiom_set.iter() {
        if let Axiom::InitiationAxiom { fluent, conditions, .. } = axiom {
            for cond in conditions {
                if let AxiomCondition::FluentHolds(fid) = cond {
                    if fid == fluent {
                        warnings.push(format!(
                            "Axiom '{}': fluent {} both initiated and required in condition (potential circular dependency)",
                            axiom.name(), fluent
                        ));
                    }
                }
            }
        }
    }

    // Check for contradictory state constraints
    let mut constraint_map: HashMap<FluentId, Vec<bool>> = HashMap::new();
    for (_, axiom) in domain.axiom_set.iter() {
        if let Axiom::StateConstraint { fluent, required_value, .. } = axiom {
            constraint_map
                .entry(*fluent)
                .or_default()
                .push(required_value.holds());
        }
    }
    for (fid, vals) in &constraint_map {
        if vals.contains(&true) && vals.contains(&false) {
            warnings.push(format!(
                "Fluent {} has contradictory state constraints (both true and false required)",
                fid
            ));
        }
    }

    // Check for unreferenced fluents
    let referenced = domain.axiom_set.all_referenced_fluents();
    for fid in domain.fluent_store.ids() {
        if !referenced.contains(&fid) {
            warnings.push(format!("Fluent {} is declared but never referenced in axioms", fid));
        }
    }

    // Validate axiom set internal consistency
    let axiom_warnings = validate_axiom_set(&domain.axiom_set);
    warnings.extend(axiom_warnings);

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grab_release_domain() {
        let domain = XRDomain::grab_release();
        assert!(domain.axiom_count() >= 3);
        assert!(domain.fluent_count() >= 3);
        assert!(!domain.spatial_predicates.is_empty());
    }

    #[test]
    fn test_gaze_dwell_domain() {
        let domain = XRDomain::gaze_dwell();
        assert!(domain.axiom_count() >= 3);
        assert!(domain.fluent_count() >= 3);
    }

    #[test]
    fn test_proximity_trigger_domain() {
        let domain = XRDomain::proximity_trigger();
        assert!(domain.axiom_count() >= 3);
        assert!(!domain.spatial_predicates.is_empty());
    }

    #[test]
    fn test_hand_menu_domain() {
        let domain = XRDomain::hand_menu();
        assert!(domain.axiom_count() >= 4);
        assert!(domain.fluent_count() >= 4);
    }

    #[test]
    fn test_two_hand_manipulation_domain() {
        let domain = XRDomain::two_hand_manipulation();
        assert!(domain.axiom_count() >= 4);
        assert!(domain.fluent_count() >= 5);
    }

    #[test]
    fn test_domain_builder() {
        let builder = DomainBuilder::new("custom");
        let (builder, f1) = builder.add_boolean_fluent("active", false);
        let (builder, f2) = builder.add_boolean_fluent("ready", true);
        let builder = builder.add_initiation(
            "activate",
            EventPattern::ActionMatch(ActionType::Activate),
            f1,
            Fluent::boolean("active", true),
            vec![AxiomCondition::FluentHolds(f2)],
        );
        let builder = builder.add_termination(
            "deactivate",
            EventPattern::ActionMatch(ActionType::Deactivate),
            f1,
            vec![],
        );
        let domain = builder.build().unwrap();
        assert_eq!(domain.axiom_count(), 2);
        assert_eq!(domain.fluent_count(), 2);
    }

    #[test]
    fn test_domain_validation_clean() {
        let domain = XRDomain::grab_release();
        let warnings = validate_domain(&domain);
        // Grab/release domain should have no critical warnings
        let circular: Vec<_> = warnings.iter().filter(|w| w.contains("circular")).collect();
        assert!(circular.is_empty());
    }

    #[test]
    fn test_domain_validation_contradiction() {
        let mut domain = XRDomain::new("bad");
        let fid = domain.fluent_store.insert(Fluent::boolean("x", false));

        domain.axiom_set.add(Axiom::StateConstraint {
            id: AxiomId(1),
            name: "force_true".into(),
            conditions: vec![],
            fluent: fid,
            required_value: Fluent::boolean("x", true),
        });
        domain.axiom_set.add(Axiom::StateConstraint {
            id: AxiomId(2),
            name: "force_false".into(),
            conditions: vec![],
            fluent: fid,
            required_value: Fluent::boolean("x", false),
        });

        let warnings = validate_domain(&domain);
        assert!(warnings.iter().any(|w| w.contains("contradictory")));
    }

    #[test]
    fn test_domain_builder_with_spatial() {
        let builder = DomainBuilder::new("spatial_test");
        let (builder, sp_id) = builder.add_spatial_predicate(SpatialPredicate::Near {
            a: EntityId(1),
            b: EntityId(2),
            distance: 1.0,
        });
        let (builder, fid) = builder.add_boolean_fluent("near", false);
        let builder = builder.add_initiation(
            "became_near",
            EventPattern::SpatialChangeMatch(sp_id),
            fid,
            Fluent::boolean("near", true),
            vec![],
        );
        let domain = builder.build().unwrap();
        assert_eq!(domain.spatial_predicates.len(), 1);
    }

    #[test]
    fn test_domain_builder_with_timer() {
        let builder = DomainBuilder::new("timer_test");
        let (builder, _timer_id) = builder.add_timer_fluent("countdown");
        let (builder, active_id) = builder.add_boolean_fluent("active", false);
        let builder = builder.add_initiation(
            "timer_done",
            EventPattern::KindMatch(EventKind::TimerExpired { name: "countdown".into() }),
            active_id,
            Fluent::boolean("active", true),
            vec![],
        );
        let domain = builder.build().unwrap();
        assert_eq!(domain.fluent_count(), 2);
    }

    #[test]
    fn test_domain_builder_causal() {
        let builder = DomainBuilder::new("causal_test");
        let (builder, _) = builder.add_boolean_fluent("triggered", false);
        let builder = builder.add_causal(
            "cause_effect",
            EventPattern::GestureMatch(GestureType::Tap),
            EventKind::Custom { name: "feedback".into(), params: HashMap::new() },
            vec![],
        );
        let domain = builder.build().unwrap();
        assert_eq!(domain.axiom_count(), 1);
    }
}

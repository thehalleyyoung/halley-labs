//! Abstract transfer functions for pharmacokinetic abstract interpretation.
//!
//! Transfer functions model the effect of medication actions (start, stop,
//! dose adjustment, PK evolution, enzyme inhibition) on the abstract state.
//! Each transfer is applied during fixed-point iteration to propagate abstract
//! values along PTA edges.
//!
//! # Transfer hierarchy
//!
//! ```text
//! TransferFunction (trait)
//! ├── MedicationStartTransfer    — introduce a drug's steady-state interval
//! ├── MedicationStopTransfer     — remove a drug (decay to zero)
//! ├── DoseAdjustTransfer         — scale concentration by dose ratio
//! ├── PkEvolutionTransfer        — time-based concentration decay
//! ├── EnzymeInhibitionTransfer   — reduce enzyme activity
//! └── CompositeTransfer          — sequential composition of transfers
//!
//! AbstractGuardEval              — evaluate guards on abstract states
//! ```

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::domain::{
    AbstractValue, BoolAbstractValue, ClinicalAbstractDomain,
    ConcentrationAbstractDomain, ConcentrationInterval, CypEnzyme, DrugId,
    EnzymeAbstractDomain, EnzymeActivityAbstractInterval, InhibitionEffect,
    InhibitionType, InductionEffect, PkParameters, DosingSchedule, ProductDomain,
    TherapeuticWindow,
};

// ============================================================================
// TransferFunction trait
// ============================================================================

/// An abstract transfer function that transforms abstract states.
///
/// Transfer functions model the semantics of PTA edges — each edge action
/// produces a new abstract state by over-approximating all concrete
/// transformations representable by the action.
pub trait TransferFunction: fmt::Debug {
    /// Apply this transfer to an abstract state, producing a new state.
    fn apply(&self, state: &ProductDomain) -> ProductDomain;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;
}

// ============================================================================
// MedicationAction — shared action descriptor
// ============================================================================

/// Describes a medication action that a transfer function models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MedicationAction {
    Start {
        drug: DrugId,
        dose_mg: f64,
        interval_hours: f64,
        pk: PkParameters,
        therapeutic_window: Option<TherapeuticWindow>,
    },
    Stop {
        drug: DrugId,
        half_life_hours: f64,
    },
    AdjustDose {
        drug: DrugId,
        old_dose_mg: f64,
        new_dose_mg: f64,
    },
    TimeElapse {
        drug: DrugId,
        hours: f64,
        elimination_rate: f64,
    },
    Inhibit {
        inhibitor: DrugId,
        effect: InhibitionEffect,
    },
    Induce {
        inducer: DrugId,
        effect: InductionEffect,
    },
}

impl fmt::Display for MedicationAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MedicationAction::Start { drug, dose_mg, .. } => {
                write!(f, "Start({}, {:.1}mg)", drug, dose_mg)
            }
            MedicationAction::Stop { drug, .. } => write!(f, "Stop({})", drug),
            MedicationAction::AdjustDose { drug, new_dose_mg, .. } => {
                write!(f, "AdjustDose({}, {:.1}mg)", drug, new_dose_mg)
            }
            MedicationAction::TimeElapse { drug, hours, .. } => {
                write!(f, "TimeElapse({}, {:.1}h)", drug, hours)
            }
            MedicationAction::Inhibit { inhibitor, effect } => {
                write!(f, "Inhibit({}, {})", inhibitor, effect.enzyme)
            }
            MedicationAction::Induce { inducer, effect } => {
                write!(f, "Induce({}, {})", inducer, effect.enzyme)
            }
        }
    }
}

// ============================================================================
// MedicationStartTransfer
// ============================================================================

/// Transfer function for starting a new medication.
///
/// Computes the expected steady-state concentration interval from PK
/// parameters and adds it to the abstract state. If a therapeutic window
/// is provided, it is intersected with the PK-derived interval for precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationStartTransfer {
    pub drug: DrugId,
    pub dose_mg: f64,
    pub interval_hours: f64,
    pub pk: PkParameters,
    pub therapeutic_window: Option<TherapeuticWindow>,
    /// Population variability factor (multiplier on Css for upper bound).
    pub variability_factor: f64,
}

impl MedicationStartTransfer {
    pub fn new(drug: DrugId, dose_mg: f64, interval_hours: f64, pk: PkParameters) -> Self {
        MedicationStartTransfer {
            drug,
            dose_mg,
            interval_hours,
            pk,
            therapeutic_window: None,
            variability_factor: 1.5,
        }
    }

    pub fn with_therapeutic_window(mut self, tw: TherapeuticWindow) -> Self {
        self.therapeutic_window = Some(tw);
        self
    }

    pub fn with_variability(mut self, factor: f64) -> Self {
        self.variability_factor = factor.max(1.0);
        self
    }

    /// Compute the expected steady-state concentration interval.
    fn compute_css_interval(&self) -> ConcentrationInterval {
        let schedule = DosingSchedule::new(self.dose_mg, self.interval_hours);
        let css_avg = self.pk.steady_state_concentration(&schedule);
        let css_max = self.pk.steady_state_cmax(&schedule);

        let lo = (css_avg / self.variability_factor).max(0.0);
        let hi = css_max * self.variability_factor;

        if lo > hi {
            ConcentrationInterval::new(0.0, hi)
        } else {
            ConcentrationInterval::new(lo, hi)
        }
    }
}

impl TransferFunction for MedicationStartTransfer {
    fn apply(&self, state: &ProductDomain) -> ProductDomain {
        let mut result = state.clone();
        let css_interval = self.compute_css_interval();

        // Join with existing concentration (if re-starting a drug)
        let existing = result.concentrations.get(&self.drug);
        let new_interval = if existing.is_bottom() {
            css_interval
        } else {
            existing.join(&css_interval)
        };

        result.concentrations.set(self.drug.clone(), new_interval);
        result.clinical.add_condition(format!("active:{}", self.drug));
        result
    }

    fn name(&self) -> &str {
        "MedicationStart"
    }
}

// ============================================================================
// MedicationStopTransfer
// ============================================================================

/// Transfer function for stopping a medication.
///
/// Models drug washout by scaling the current concentration interval toward
/// zero based on the drug's half-life. After a sufficient number of
/// half-lives, the interval collapses to [0, residual].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationStopTransfer {
    pub drug: DrugId,
    pub half_life_hours: f64,
    /// Number of half-lives to model for washout (default: 5).
    pub washout_half_lives: f64,
}

impl MedicationStopTransfer {
    pub fn new(drug: DrugId, half_life_hours: f64) -> Self {
        MedicationStopTransfer {
            drug,
            half_life_hours: half_life_hours.max(0.01),
            washout_half_lives: 5.0,
        }
    }

    pub fn with_washout_half_lives(mut self, n: f64) -> Self {
        self.washout_half_lives = n.max(1.0);
        self
    }

    fn washout_fraction(&self) -> f64 {
        let total_time = self.washout_half_lives * self.half_life_hours;
        let ke = 0.693 / self.half_life_hours;
        (-ke * total_time).exp()
    }
}

impl TransferFunction for MedicationStopTransfer {
    fn apply(&self, state: &ProductDomain) -> ProductDomain {
        let mut result = state.clone();
        let current = result.concentrations.get(&self.drug);

        if current.is_bottom() {
            return result;
        }

        let fraction = self.washout_fraction();
        let new_interval = ConcentrationInterval::new(
            0.0,
            (current.hi * fraction).max(0.0),
        );

        if new_interval.hi < 1e-10 {
            result.concentrations.set(self.drug.clone(), ConcentrationInterval::bottom());
        } else {
            result.concentrations.set(self.drug.clone(), new_interval);
        }
        result
    }

    fn name(&self) -> &str {
        "MedicationStop"
    }
}

// ============================================================================
// DoseAdjustTransfer
// ============================================================================

/// Transfer function for adjusting a medication dose.
///
/// Scales the current concentration interval by the ratio of new to old dose,
/// which is sound because steady-state concentration is proportional to dose
/// in linear PK.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseAdjustTransfer {
    pub drug: DrugId,
    pub old_dose_mg: f64,
    pub new_dose_mg: f64,
    /// Transition uncertainty factor (widens interval during transition).
    pub transition_factor: f64,
}

impl DoseAdjustTransfer {
    pub fn new(drug: DrugId, old_dose: f64, new_dose: f64) -> Self {
        DoseAdjustTransfer {
            drug,
            old_dose_mg: old_dose.max(f64::EPSILON),
            new_dose_mg: new_dose.max(0.0),
            transition_factor: 1.2,
        }
    }

    pub fn with_transition_factor(mut self, factor: f64) -> Self {
        self.transition_factor = factor.max(1.0);
        self
    }

    fn dose_ratio(&self) -> f64 {
        self.new_dose_mg / self.old_dose_mg
    }
}

impl TransferFunction for DoseAdjustTransfer {
    fn apply(&self, state: &ProductDomain) -> ProductDomain {
        let mut result = state.clone();
        let current = result.concentrations.get(&self.drug);

        if current.is_bottom() {
            return result;
        }

        let ratio = self.dose_ratio();
        let target = current.scale(ratio);

        // During transition, the interval spans from the minimum of old and
        // new to the maximum, widened by the transition factor.
        let lo = target.lo.min(current.lo) / self.transition_factor;
        let hi = target.hi.max(current.hi) * self.transition_factor;

        result.concentrations.set(
            self.drug.clone(),
            ConcentrationInterval::new(lo.max(0.0), hi),
        );
        result
    }

    fn name(&self) -> &str {
        "DoseAdjust"
    }
}

// ============================================================================
// PkEvolutionTransfer
// ============================================================================

/// Transfer function modelling time-based pharmacokinetic evolution.
///
/// Computes the interval concentration change over a time period using
/// first-order elimination kinetics. The concentration decays exponentially
/// according to the drug's elimination rate constant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkEvolutionTransfer {
    pub drug: DrugId,
    pub hours: f64,
    pub elimination_rate: f64,
    /// Optional dosing to model periodic absorption.
    pub dosing: Option<DosingSchedule>,
    pub volume_of_distribution: f64,
}

impl PkEvolutionTransfer {
    pub fn new(drug: DrugId, hours: f64, elimination_rate: f64) -> Self {
        PkEvolutionTransfer {
            drug,
            hours: hours.max(0.0),
            elimination_rate: elimination_rate.max(0.0),
            dosing: None,
            volume_of_distribution: 70.0,
        }
    }

    pub fn with_dosing(mut self, schedule: DosingSchedule, vd: f64) -> Self {
        self.dosing = Some(schedule);
        self.volume_of_distribution = vd.max(f64::EPSILON);
        self
    }

    /// Fraction remaining after elimination over the time period.
    fn decay_fraction(&self) -> f64 {
        (-self.elimination_rate * self.hours).exp()
    }

    /// Absorption contribution from periodic dosing during the time period.
    fn absorption_contribution(&self) -> ConcentrationInterval {
        match &self.dosing {
            None => ConcentrationInterval::new(0.0, 0.0),
            Some(schedule) => {
                if schedule.interval_hours <= 0.0 || self.volume_of_distribution <= 0.0 {
                    return ConcentrationInterval::new(0.0, 0.0);
                }
                let doses_in_period = (self.hours / schedule.interval_hours).ceil() as usize;
                if doses_in_period == 0 {
                    return ConcentrationInterval::new(0.0, 0.0);
                }
                let per_dose = schedule.effective_dose() / self.volume_of_distribution;
                let min_absorbed = per_dose * 0.8;
                let max_absorbed = per_dose * 1.2 * doses_in_period as f64;
                ConcentrationInterval::new(min_absorbed.max(0.0), max_absorbed)
            }
        }
    }
}

impl TransferFunction for PkEvolutionTransfer {
    fn apply(&self, state: &ProductDomain) -> ProductDomain {
        let mut result = state.clone();
        let current = result.concentrations.get(&self.drug);

        if current.is_bottom() {
            return result;
        }

        let decay = self.decay_fraction();
        let decayed = ConcentrationInterval::new(
            (current.lo * decay).max(0.0),
            current.hi * decay,
        );

        let absorbed = self.absorption_contribution();
        let new_interval = decayed.add(&absorbed);

        let final_interval = ConcentrationInterval::new(
            new_interval.lo.max(0.0),
            new_interval.hi,
        );

        result.concentrations.set(self.drug.clone(), final_interval);
        result
    }

    fn name(&self) -> &str {
        "PkEvolution"
    }
}

// ============================================================================
// EnzymeInhibitionTransfer
// ============================================================================

/// Transfer function modelling CYP enzyme inhibition by a perpetrator drug.
///
/// Reduces the activity interval of the target enzyme based on the
/// inhibitor's concentration and Ki value. Also scales the concentration
/// of victim drugs metabolised through the inhibited enzyme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeInhibitionTransfer {
    pub inhibitor: DrugId,
    pub effect: InhibitionEffect,
    /// Drugs metabolised through the inhibited enzyme, with fraction metabolised.
    pub victim_drugs: Vec<(DrugId, f64)>,
}

impl EnzymeInhibitionTransfer {
    pub fn new(inhibitor: DrugId, effect: InhibitionEffect) -> Self {
        EnzymeInhibitionTransfer {
            inhibitor,
            effect,
            victim_drugs: Vec::new(),
        }
    }

    pub fn with_victim(mut self, drug: DrugId, fraction_metabolized: f64) -> Self {
        self.victim_drugs.push((drug, fraction_metabolized.clamp(0.0, 1.0)));
        self
    }

    /// Compute the enzyme activity reduction interval given inhibitor concentration.
    fn compute_activity_interval(
        &self,
        inhibitor_conc: &ConcentrationInterval,
    ) -> EnzymeActivityAbstractInterval {
        if inhibitor_conc.is_bottom() {
            return EnzymeActivityAbstractInterval::normal();
        }

        // AUC ratio = 1 + [I]/Ki for competitive inhibition
        // Activity fold = 1 / AUC_ratio
        let auc_ratio_lo = self.effect.auc_ratio(inhibitor_conc.lo.max(0.0));
        let auc_ratio_hi = self.effect.auc_ratio(inhibitor_conc.hi.max(0.0));

        let activity_hi = (1.0 / auc_ratio_lo).min(1.0);
        let activity_lo = (1.0 / auc_ratio_hi).max(0.0);

        EnzymeActivityAbstractInterval::new(activity_lo, activity_hi)
    }

    /// Compute the concentration scaling factor for victim drugs.
    fn victim_auc_ratio_interval(
        &self,
        inhibitor_conc: &ConcentrationInterval,
        fraction_metabolized: f64,
    ) -> ConcentrationInterval {
        if inhibitor_conc.is_bottom() {
            return ConcentrationInterval::new(1.0, 1.0);
        }

        let auc_lo = self.effect.auc_ratio(inhibitor_conc.lo.max(0.0));
        let auc_hi = self.effect.auc_ratio(inhibitor_conc.hi.max(0.0));

        // Net AUC change = 1 / (1 - fm + fm/AUC_ratio) ... simplified to
        // scaling ≈ 1 + fm * (AUC_ratio - 1)
        let scale_lo = 1.0 + fraction_metabolized * (auc_lo - 1.0);
        let scale_hi = 1.0 + fraction_metabolized * (auc_hi - 1.0);

        ConcentrationInterval::new(scale_lo.max(1.0), scale_hi.max(1.0))
    }
}

impl TransferFunction for EnzymeInhibitionTransfer {
    fn apply(&self, state: &ProductDomain) -> ProductDomain {
        let mut result = state.clone();
        let inhibitor_conc = result.concentrations.get(&self.inhibitor);

        // Update enzyme activity
        let new_activity = self.compute_activity_interval(&inhibitor_conc);
        let current_activity = result.enzymes.get(&self.effect.enzyme);
        let combined = current_activity.meet(&new_activity);
        result.enzymes.set(self.effect.enzyme, combined);

        // Update victim drug concentrations
        for (victim, fm) in &self.victim_drugs {
            let victim_conc = result.concentrations.get(victim);
            if victim_conc.is_bottom() {
                continue;
            }
            let scaling = self.victim_auc_ratio_interval(&inhibitor_conc, *fm);
            let new_conc = victim_conc.mul(&scaling);
            result.concentrations.set(victim.clone(), new_conc);
        }

        result.clinical.add_condition(format!(
            "inhibition:{}->{}",
            self.inhibitor, self.effect.enzyme
        ));
        result
    }

    fn name(&self) -> &str {
        "EnzymeInhibition"
    }
}

// ============================================================================
// EnzymeInductionTransfer
// ============================================================================

/// Transfer function modelling CYP enzyme induction by a perpetrator drug.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeInductionTransfer {
    pub inducer: DrugId,
    pub effect: InductionEffect,
    pub victim_drugs: Vec<(DrugId, f64)>,
}

impl EnzymeInductionTransfer {
    pub fn new(inducer: DrugId, effect: InductionEffect) -> Self {
        EnzymeInductionTransfer {
            inducer,
            effect,
            victim_drugs: Vec::new(),
        }
    }

    pub fn with_victim(mut self, drug: DrugId, fraction_metabolized: f64) -> Self {
        self.victim_drugs.push((drug, fraction_metabolized.clamp(0.0, 1.0)));
        self
    }

    fn compute_induction_interval(
        &self,
        inducer_conc: &ConcentrationInterval,
    ) -> EnzymeActivityAbstractInterval {
        if inducer_conc.is_bottom() {
            return EnzymeActivityAbstractInterval::normal();
        }
        let fold_lo = self.effect.fold_induction(inducer_conc.lo.max(0.0));
        let fold_hi = self.effect.fold_induction(inducer_conc.hi.max(0.0));
        EnzymeActivityAbstractInterval::new(fold_lo.min(fold_hi), fold_lo.max(fold_hi))
    }
}

impl TransferFunction for EnzymeInductionTransfer {
    fn apply(&self, state: &ProductDomain) -> ProductDomain {
        let mut result = state.clone();
        let inducer_conc = result.concentrations.get(&self.inducer);

        let induction = self.compute_induction_interval(&inducer_conc);
        let current = result.enzymes.get(&self.effect.enzyme);
        let combined = current.join(&induction);
        result.enzymes.set(self.effect.enzyme, combined);

        // Induction increases clearance, decreasing victim concentrations
        for (victim, fm) in &self.victim_drugs {
            let victim_conc = result.concentrations.get(victim);
            if victim_conc.is_bottom() {
                continue;
            }
            let fold_lo = self.effect.fold_induction(inducer_conc.lo.max(0.0));
            let fold_hi = self.effect.fold_induction(inducer_conc.hi.max(0.0));
            let reduction_hi = 1.0 / (1.0 + fm * (fold_lo.min(fold_hi) - 1.0));
            let reduction_lo = 1.0 / (1.0 + fm * (fold_lo.max(fold_hi) - 1.0));
            let scale = ConcentrationInterval::new(
                reduction_lo.clamp(0.0, 1.0),
                reduction_hi.clamp(0.0, 1.0),
            );
            let new_conc = victim_conc.mul(&scale);
            result.concentrations.set(victim.clone(), new_conc);
        }

        result
    }

    fn name(&self) -> &str {
        "EnzymeInduction"
    }
}

// ============================================================================
// AbstractGuardEval
// ============================================================================

/// Evaluator for abstract guards on abstract states.
///
/// Guards express conditions like "concentration of drug X ≤ threshold" or
/// "enzyme activity of CYP3A4 ≥ 0.5". The evaluator returns a three-valued
/// boolean indicating whether the guard is definitely true, definitely false,
/// or unknown on the abstract state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AbstractGuard {
    /// Drug concentration is within a range.
    ConcentrationInRange {
        drug: DrugId,
        lo: f64,
        hi: f64,
    },
    /// Drug concentration exceeds a threshold.
    ConcentrationAbove {
        drug: DrugId,
        threshold: f64,
    },
    /// Drug concentration is below a threshold.
    ConcentrationBelow {
        drug: DrugId,
        threshold: f64,
    },
    /// Enzyme activity is within a range.
    EnzymeActivityInRange {
        enzyme: CypEnzyme,
        lo: f64,
        hi: f64,
    },
    /// Clinical condition is present.
    ConditionPresent {
        condition: String,
    },
    /// Conjunction of guards.
    And(Vec<AbstractGuard>),
    /// Disjunction of guards.
    Or(Vec<AbstractGuard>),
    /// Negation of a guard.
    Not(Box<AbstractGuard>),
    /// Always true.
    True,
    /// Always false.
    False,
}

/// Evaluates abstract guards on abstract states.
#[derive(Debug, Clone)]
pub struct AbstractGuardEval;

impl AbstractGuardEval {
    pub fn new() -> Self {
        AbstractGuardEval
    }

    /// Evaluate a guard on an abstract state.
    pub fn evaluate(&self, guard: &AbstractGuard, state: &ProductDomain) -> BoolAbstractValue {
        match guard {
            AbstractGuard::True => BoolAbstractValue::True,
            AbstractGuard::False => BoolAbstractValue::False,

            AbstractGuard::ConcentrationInRange { drug, lo, hi } => {
                let interval = state.concentrations.get(drug);
                if interval.is_bottom() {
                    return BoolAbstractValue::Bottom;
                }
                let range = ConcentrationInterval::new(*lo, *hi);
                if range.contains_interval(&interval) {
                    BoolAbstractValue::True
                } else if !interval.overlaps(&range) {
                    BoolAbstractValue::False
                } else {
                    BoolAbstractValue::Top
                }
            }

            AbstractGuard::ConcentrationAbove { drug, threshold } => {
                let interval = state.concentrations.get(drug);
                if interval.is_bottom() {
                    return BoolAbstractValue::Bottom;
                }
                if interval.lo > *threshold {
                    BoolAbstractValue::True
                } else if interval.hi <= *threshold {
                    BoolAbstractValue::False
                } else {
                    BoolAbstractValue::Top
                }
            }

            AbstractGuard::ConcentrationBelow { drug, threshold } => {
                let interval = state.concentrations.get(drug);
                if interval.is_bottom() {
                    return BoolAbstractValue::Bottom;
                }
                if interval.hi < *threshold {
                    BoolAbstractValue::True
                } else if interval.lo >= *threshold {
                    BoolAbstractValue::False
                } else {
                    BoolAbstractValue::Top
                }
            }

            AbstractGuard::EnzymeActivityInRange { enzyme, lo, hi } => {
                let activity = state.enzymes.get(enzyme);
                if activity.is_bottom() {
                    return BoolAbstractValue::Bottom;
                }
                if activity.lo >= *lo && activity.hi <= *hi {
                    BoolAbstractValue::True
                } else if activity.hi < *lo || activity.lo > *hi {
                    BoolAbstractValue::False
                } else {
                    BoolAbstractValue::Top
                }
            }

            AbstractGuard::ConditionPresent { condition } => {
                if state.clinical.conditions.is_top {
                    BoolAbstractValue::Top
                } else if state.clinical.has_condition(condition) {
                    BoolAbstractValue::True
                } else {
                    BoolAbstractValue::Top
                }
            }

            AbstractGuard::And(guards) => {
                let mut result = BoolAbstractValue::True;
                for g in guards {
                    let v = self.evaluate(g, state);
                    result = result.abstract_and(&v);
                }
                result
            }

            AbstractGuard::Or(guards) => {
                let mut result = BoolAbstractValue::False;
                for g in guards {
                    let v = self.evaluate(g, state);
                    result = result.abstract_or(&v);
                }
                result
            }

            AbstractGuard::Not(inner) => {
                self.evaluate(inner, state).negate()
            }
        }
    }
}

impl Default for AbstractGuardEval {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CompositeTransfer
// ============================================================================

/// Sequential composition of multiple transfer functions.
///
/// Applies each constituent transfer in order, threading the abstract state
/// through the chain. This models PTA edges that perform multiple
/// simultaneous actions.
#[derive(Debug)]
pub struct CompositeTransfer {
    pub transfers: Vec<Box<dyn TransferFunction>>,
    pub label: String,
}

impl CompositeTransfer {
    pub fn new(label: impl Into<String>) -> Self {
        CompositeTransfer {
            transfers: Vec::new(),
            label: label.into(),
        }
    }

    pub fn add(mut self, transfer: Box<dyn TransferFunction>) -> Self {
        self.transfers.push(transfer);
        self
    }

    pub fn len(&self) -> usize {
        self.transfers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transfers.is_empty()
    }
}

impl TransferFunction for CompositeTransfer {
    fn apply(&self, state: &ProductDomain) -> ProductDomain {
        let mut current = state.clone();
        for transfer in &self.transfers {
            current = transfer.apply(&current);
        }
        current
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ============================================================================
// IdentityTransfer
// ============================================================================

/// No-op transfer that returns the state unchanged.
#[derive(Debug, Clone)]
pub struct IdentityTransfer;

impl TransferFunction for IdentityTransfer {
    fn apply(&self, state: &ProductDomain) -> ProductDomain {
        state.clone()
    }

    fn name(&self) -> &str {
        "Identity"
    }
}

// ============================================================================
// TransferResult — enriched transfer output
// ============================================================================

/// Result of applying a transfer, including diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferResult {
    pub new_state: ProductDomain,
    pub drugs_affected: Vec<DrugId>,
    pub enzymes_affected: Vec<CypEnzyme>,
    pub description: String,
}

impl TransferResult {
    pub fn new(state: ProductDomain, description: impl Into<String>) -> Self {
        TransferResult {
            new_state: state,
            drugs_affected: Vec::new(),
            enzymes_affected: Vec::new(),
            description: description.into(),
        }
    }

    pub fn with_drug(mut self, drug: DrugId) -> Self {
        self.drugs_affected.push(drug);
        self
    }

    pub fn with_enzyme(mut self, enzyme: CypEnzyme) -> Self {
        self.enzymes_affected.push(enzyme);
        self
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pk() -> PkParameters {
        PkParameters::new(5.0, 50.0)
    }

    fn make_initial_state() -> ProductDomain {
        ProductDomain::initial()
    }

    fn state_with_drug(drug: &str, lo: f64, hi: f64) -> ProductDomain {
        let mut state = ProductDomain::initial();
        state.set_concentration(DrugId::new(drug), ConcentrationInterval::new(lo, hi));
        state
    }

    // ------------------------------------------------------------------
    // MedicationStartTransfer
    // ------------------------------------------------------------------

    #[test]
    fn test_medication_start_basic() {
        let transfer = MedicationStartTransfer::new(
            DrugId::new("warfarin"),
            5.0,
            24.0,
            make_pk(),
        );
        let state = make_initial_state();
        let result = transfer.apply(&state);
        let conc = result.concentrations.get(&DrugId::new("warfarin"));
        assert!(!conc.is_bottom());
        assert!(conc.lo >= 0.0);
        assert!(conc.hi > conc.lo);
    }

    #[test]
    fn test_medication_start_adds_condition() {
        let transfer = MedicationStartTransfer::new(
            DrugId::new("aspirin"),
            100.0,
            8.0,
            make_pk(),
        );
        let state = make_initial_state();
        let result = transfer.apply(&state);
        assert!(result.clinical.has_condition("active:aspirin"));
    }

    #[test]
    fn test_medication_start_joins_existing() {
        let transfer = MedicationStartTransfer::new(
            DrugId::new("drug_a"),
            10.0,
            12.0,
            make_pk(),
        );
        let state = state_with_drug("drug_a", 0.5, 1.5);
        let result = transfer.apply(&state);
        let conc = result.concentrations.get(&DrugId::new("drug_a"));
        assert!(conc.lo <= 0.5);
        assert!(conc.hi >= 1.5);
    }

    // ------------------------------------------------------------------
    // MedicationStopTransfer
    // ------------------------------------------------------------------

    #[test]
    fn test_medication_stop() {
        let transfer = MedicationStopTransfer::new(DrugId::new("drug_a"), 6.0);
        let state = state_with_drug("drug_a", 2.0, 8.0);
        let result = transfer.apply(&state);
        let conc = result.concentrations.get(&DrugId::new("drug_a"));
        assert!(conc.lo < 2.0);
        assert!(conc.hi < 8.0);
    }

    #[test]
    fn test_medication_stop_on_absent_drug() {
        let transfer = MedicationStopTransfer::new(DrugId::new("absent"), 6.0);
        let state = make_initial_state();
        let result = transfer.apply(&state);
        let conc = result.concentrations.get(&DrugId::new("absent"));
        assert!(conc.is_bottom());
    }

    // ------------------------------------------------------------------
    // DoseAdjustTransfer
    // ------------------------------------------------------------------

    #[test]
    fn test_dose_adjust_increase() {
        let transfer = DoseAdjustTransfer::new(DrugId::new("drug_a"), 5.0, 10.0);
        let state = state_with_drug("drug_a", 1.0, 3.0);
        let result = transfer.apply(&state);
        let conc = result.concentrations.get(&DrugId::new("drug_a"));
        // Doubling dose should roughly double the concentration
        assert!(conc.hi > 3.0);
    }

    #[test]
    fn test_dose_adjust_decrease() {
        let transfer = DoseAdjustTransfer::new(DrugId::new("drug_a"), 10.0, 5.0);
        let state = state_with_drug("drug_a", 2.0, 6.0);
        let result = transfer.apply(&state);
        let conc = result.concentrations.get(&DrugId::new("drug_a"));
        assert!(conc.lo < 2.0);
    }

    // ------------------------------------------------------------------
    // PkEvolutionTransfer
    // ------------------------------------------------------------------

    #[test]
    fn test_pk_evolution_decay() {
        let ke = 0.693 / 6.0; // half-life 6h
        let transfer = PkEvolutionTransfer::new(DrugId::new("drug_a"), 6.0, ke);
        let state = state_with_drug("drug_a", 4.0, 8.0);
        let result = transfer.apply(&state);
        let conc = result.concentrations.get(&DrugId::new("drug_a"));
        // After one half-life, concentrations should roughly halve
        assert!(conc.hi < 8.0);
        assert!((conc.hi - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_pk_evolution_with_dosing() {
        let ke = 0.693 / 6.0;
        let schedule = DosingSchedule::new(100.0, 8.0);
        let transfer = PkEvolutionTransfer::new(DrugId::new("drug_a"), 8.0, ke)
            .with_dosing(schedule, 50.0);
        let state = state_with_drug("drug_a", 1.0, 3.0);
        let result = transfer.apply(&state);
        let conc = result.concentrations.get(&DrugId::new("drug_a"));
        // With dosing, concentration should not just decay
        assert!(conc.hi > 0.0);
    }

    // ------------------------------------------------------------------
    // EnzymeInhibitionTransfer
    // ------------------------------------------------------------------

    #[test]
    fn test_enzyme_inhibition() {
        let effect = InhibitionEffect::new(CypEnzyme::CYP3A4, InhibitionType::Competitive, 1.0);
        let transfer = EnzymeInhibitionTransfer::new(DrugId::new("inhibitor"), effect)
            .with_victim(DrugId::new("victim"), 0.8);

        let mut state = state_with_drug("inhibitor", 5.0, 10.0);
        state.set_concentration(DrugId::new("victim"), ConcentrationInterval::new(1.0, 3.0));

        let result = transfer.apply(&state);

        // Enzyme activity should decrease
        let activity = result.enzymes.get(&CypEnzyme::CYP3A4);
        assert!(activity.hi <= 1.0);

        // Victim concentration should increase
        let victim_conc = result.concentrations.get(&DrugId::new("victim"));
        assert!(victim_conc.hi >= 3.0);
    }

    // ------------------------------------------------------------------
    // AbstractGuardEval
    // ------------------------------------------------------------------

    #[test]
    fn test_guard_concentration_in_range_true() {
        let state = state_with_drug("drug_a", 2.0, 4.0);
        let guard = AbstractGuard::ConcentrationInRange {
            drug: DrugId::new("drug_a"),
            lo: 1.0,
            hi: 5.0,
        };
        let eval = AbstractGuardEval::new();
        assert_eq!(eval.evaluate(&guard, &state), BoolAbstractValue::True);
    }

    #[test]
    fn test_guard_concentration_in_range_false() {
        let state = state_with_drug("drug_a", 6.0, 8.0);
        let guard = AbstractGuard::ConcentrationInRange {
            drug: DrugId::new("drug_a"),
            lo: 1.0,
            hi: 5.0,
        };
        let eval = AbstractGuardEval::new();
        assert_eq!(eval.evaluate(&guard, &state), BoolAbstractValue::False);
    }

    #[test]
    fn test_guard_concentration_in_range_unknown() {
        let state = state_with_drug("drug_a", 3.0, 7.0);
        let guard = AbstractGuard::ConcentrationInRange {
            drug: DrugId::new("drug_a"),
            lo: 1.0,
            hi: 5.0,
        };
        let eval = AbstractGuardEval::new();
        assert_eq!(eval.evaluate(&guard, &state), BoolAbstractValue::Top);
    }

    #[test]
    fn test_guard_and_or() {
        let state = state_with_drug("drug_a", 2.0, 4.0);
        let g1 = AbstractGuard::ConcentrationInRange {
            drug: DrugId::new("drug_a"),
            lo: 1.0,
            hi: 5.0,
        };
        let g2 = AbstractGuard::True;
        let and_guard = AbstractGuard::And(vec![g1.clone(), g2.clone()]);
        let eval = AbstractGuardEval::new();
        assert_eq!(eval.evaluate(&and_guard, &state), BoolAbstractValue::True);

        let g3 = AbstractGuard::False;
        let or_guard = AbstractGuard::Or(vec![g1, g3]);
        assert_eq!(eval.evaluate(&or_guard, &state), BoolAbstractValue::True);
    }

    // ------------------------------------------------------------------
    // CompositeTransfer
    // ------------------------------------------------------------------

    #[test]
    fn test_composite_transfer() {
        let t1 = MedicationStartTransfer::new(
            DrugId::new("drug_a"), 10.0, 12.0, make_pk(),
        );
        let t2 = MedicationStartTransfer::new(
            DrugId::new("drug_b"), 20.0, 8.0, make_pk(),
        );
        let composite = CompositeTransfer::new("start_both")
            .add(Box::new(t1))
            .add(Box::new(t2));

        let state = make_initial_state();
        let result = composite.apply(&state);

        let a = result.concentrations.get(&DrugId::new("drug_a"));
        let b = result.concentrations.get(&DrugId::new("drug_b"));
        assert!(!a.is_bottom());
        assert!(!b.is_bottom());
    }

    #[test]
    fn test_identity_transfer() {
        let state = state_with_drug("drug_a", 1.0, 5.0);
        let result = IdentityTransfer.apply(&state);
        assert_eq!(result, state);
    }

    // ------------------------------------------------------------------
    // EnzymeInductionTransfer
    // ------------------------------------------------------------------

    #[test]
    fn test_enzyme_induction() {
        let effect = InductionEffect::new(CypEnzyme::CYP3A4, 8.0, 5.0);
        let transfer = EnzymeInductionTransfer::new(DrugId::new("inducer"), effect)
            .with_victim(DrugId::new("victim"), 0.7);

        let mut state = state_with_drug("inducer", 5.0, 10.0);
        state.set_concentration(DrugId::new("victim"), ConcentrationInterval::new(2.0, 6.0));

        let result = transfer.apply(&state);

        let activity = result.enzymes.get(&CypEnzyme::CYP3A4);
        assert!(activity.hi > 1.0);

        let victim_conc = result.concentrations.get(&DrugId::new("victim"));
        assert!(victim_conc.hi <= 6.0);
    }
}

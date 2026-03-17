//! Abstract safety checking for pharmacokinetic verification.
//!
//! This module provides abstract interpretation–based safety analysis for
//! multi-drug pharmacokinetic states. It checks concentration bounds,
//! therapeutic windows, enzyme activity constraints, and drug–drug interaction
//! severity limits against an abstract product domain.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::domain::{
    AbstractValue, ConcentrationAbstractDomain, ConcentrationInterval, ClinicalAbstractDomain,
    CypEnzyme, DrugConfig, DrugId, EnzymeAbstractDomain, EnzymeActivityAbstractInterval,
    BoolAbstractValue, InhibitionType, ProductDomain, Severity, TherapeuticWindow,
};

// ── SafetyVerdict ──────────────────────────────────────────────────────────

/// The three-valued outcome of a safety check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyVerdict {
    /// The property is definitively satisfied.
    Safe,
    /// The property is definitively violated.
    Unsafe,
    /// The property could not be decided; the string gives a reason.
    Unknown(String),
}

impl SafetyVerdict {
    pub fn is_safe(&self) -> bool {
        matches!(self, SafetyVerdict::Safe)
    }

    pub fn is_unsafe(&self) -> bool {
        matches!(self, SafetyVerdict::Unsafe)
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, SafetyVerdict::Unknown(_))
    }

    /// Conjunction: both must be safe for the result to be safe.
    pub fn combine_and(a: &SafetyVerdict, b: &SafetyVerdict) -> SafetyVerdict {
        match (a, b) {
            (SafetyVerdict::Safe, SafetyVerdict::Safe) => SafetyVerdict::Safe,
            (SafetyVerdict::Unsafe, _) | (_, SafetyVerdict::Unsafe) => SafetyVerdict::Unsafe,
            (SafetyVerdict::Unknown(r), SafetyVerdict::Safe) => {
                SafetyVerdict::Unknown(r.clone())
            }
            (SafetyVerdict::Safe, SafetyVerdict::Unknown(r)) => {
                SafetyVerdict::Unknown(r.clone())
            }
            (SafetyVerdict::Unknown(r1), SafetyVerdict::Unknown(r2)) => {
                SafetyVerdict::Unknown(format!("{} and {}", r1, r2))
            }
        }
    }

    /// Disjunction: either being safe suffices.
    pub fn combine_or(a: &SafetyVerdict, b: &SafetyVerdict) -> SafetyVerdict {
        match (a, b) {
            (SafetyVerdict::Safe, _) | (_, SafetyVerdict::Safe) => SafetyVerdict::Safe,
            (SafetyVerdict::Unsafe, SafetyVerdict::Unsafe) => SafetyVerdict::Unsafe,
            (SafetyVerdict::Unsafe, SafetyVerdict::Unknown(r)) => {
                SafetyVerdict::Unknown(r.clone())
            }
            (SafetyVerdict::Unknown(r), SafetyVerdict::Unsafe) => {
                SafetyVerdict::Unknown(r.clone())
            }
            (SafetyVerdict::Unknown(r1), SafetyVerdict::Unknown(r2)) => {
                SafetyVerdict::Unknown(format!("{} or {}", r1, r2))
            }
        }
    }
}

// ── SafetyEvidence ─────────────────────────────────────────────────────────

/// A bundle of evidence justifying a [`SafetyVerdict`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEvidence {
    pub verdict: SafetyVerdict,
    pub property_description: String,
    pub concentration_evidence: Vec<(DrugId, ConcentrationInterval)>,
    /// Each entry is `(enzyme, lo_activity, hi_activity)`.
    pub enzyme_evidence: Vec<(CypEnzyme, f64, f64)>,
    pub details: Vec<String>,
}

impl SafetyEvidence {
    pub fn new(verdict: SafetyVerdict, description: String) -> Self {
        Self {
            verdict,
            property_description: description,
            concentration_evidence: Vec::new(),
            enzyme_evidence: Vec::new(),
            details: Vec::new(),
        }
    }

    pub fn with_concentration(mut self, drug: DrugId, interval: ConcentrationInterval) -> Self {
        self.concentration_evidence.push((drug, interval));
        self
    }

    pub fn with_enzyme(mut self, enzyme: CypEnzyme, lo: f64, hi: f64) -> Self {
        self.enzyme_evidence.push((enzyme, lo, hi));
        self
    }

    pub fn add_detail(&mut self, detail: String) {
        self.details.push(detail);
    }
}

// ── SafetyProperty ─────────────────────────────────────────────────────────

/// A declarative safety property to be checked against abstract state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyProperty {
    ConcentrationBound {
        drug: DrugId,
        lo: f64,
        hi: f64,
    },
    TherapeuticWindowCheck {
        drug: DrugId,
        window: TherapeuticWindow,
    },
    EnzymeActivityBound {
        enzyme: CypEnzyme,
        min_activity: f64,
        max_activity: f64,
    },
    InteractionSeverityLimit {
        drug_a: DrugId,
        drug_b: DrugId,
        max_severity: Severity,
    },
    Conjunction(Vec<SafetyProperty>),
    Disjunction(Vec<SafetyProperty>),
}

impl SafetyProperty {
    /// Human-readable description of this property.
    pub fn description(&self) -> String {
        match self {
            SafetyProperty::ConcentrationBound { drug, lo, hi } => {
                format!(
                    "Concentration of {} must be within [{:.4}, {:.4}]",
                    drug.as_str(),
                    lo,
                    hi
                )
            }
            SafetyProperty::TherapeuticWindowCheck { drug, window } => {
                format!(
                    "Concentration of {} must be within therapeutic window [{:.4}, {:.4}]",
                    drug.as_str(),
                    window.min_concentration,
                    window.max_concentration
                )
            }
            SafetyProperty::EnzymeActivityBound {
                enzyme,
                min_activity,
                max_activity,
            } => {
                format!(
                    "Activity of {:?} must be within [{:.4}, {:.4}]",
                    enzyme, min_activity, max_activity
                )
            }
            SafetyProperty::InteractionSeverityLimit {
                drug_a,
                drug_b,
                max_severity,
            } => {
                format!(
                    "Interaction between {} and {} must be at most {:?}",
                    drug_a.as_str(),
                    drug_b.as_str(),
                    max_severity
                )
            }
            SafetyProperty::Conjunction(props) => {
                let descs: Vec<String> = props.iter().map(|p| p.description()).collect();
                format!("All of: [{}]", descs.join("; "))
            }
            SafetyProperty::Disjunction(props) => {
                let descs: Vec<String> = props.iter().map(|p| p.description()).collect();
                format!("Any of: [{}]", descs.join("; "))
            }
        }
    }

    /// Collect every [`DrugId`] referenced (transitively) by this property.
    pub fn referenced_drugs(&self) -> Vec<DrugId> {
        match self {
            SafetyProperty::ConcentrationBound { drug, .. }
            | SafetyProperty::TherapeuticWindowCheck { drug, .. } => {
                vec![drug.clone()]
            }
            SafetyProperty::EnzymeActivityBound { .. } => vec![],
            SafetyProperty::InteractionSeverityLimit { drug_a, drug_b, .. } => {
                vec![drug_a.clone(), drug_b.clone()]
            }
            SafetyProperty::Conjunction(props) | SafetyProperty::Disjunction(props) => {
                let mut drugs: Vec<DrugId> = Vec::new();
                for p in props {
                    for d in p.referenced_drugs() {
                        if !drugs.contains(&d) {
                            drugs.push(d);
                        }
                    }
                }
                drugs
            }
        }
    }
}

// ── AbstractSafetyChecker ──────────────────────────────────────────────────

/// Checks [`SafetyProperty`] values against an abstract [`ProductDomain`].
pub struct AbstractSafetyChecker {
    drug_configs: Vec<DrugConfig>,
}

impl AbstractSafetyChecker {
    pub fn new(drug_configs: Vec<DrugConfig>) -> Self {
        Self { drug_configs }
    }

    fn find_config(&self, drug: &DrugId) -> Option<&DrugConfig> {
        self.drug_configs.iter().find(|c| c.drug_id == *drug)
    }

    /// Map an AUC ratio to an interaction severity level.
    fn severity_from_auc_ratio(auc_ratio: f64) -> Severity {
        if auc_ratio < 1.25 {
            Severity::Minor
        } else if auc_ratio < 2.0 {
            Severity::Moderate
        } else if auc_ratio < 5.0 {
            Severity::Major
        } else {
            Severity::Critical
        }
    }

    // ── Top-level dispatcher ───────────────────────────────────────────

    pub fn check_safety(
        &self,
        state: &ProductDomain,
        property: &SafetyProperty,
    ) -> SafetyEvidence {
        match property {
            SafetyProperty::ConcentrationBound { drug, lo, hi } => {
                self.check_concentration_bound(state, drug, *lo, *hi)
            }
            SafetyProperty::TherapeuticWindowCheck { drug, window } => {
                self.check_therapeutic_window(state, drug, window)
            }
            SafetyProperty::EnzymeActivityBound {
                enzyme,
                min_activity,
                max_activity,
            } => self.check_enzyme_bound(state, enzyme, *min_activity, *max_activity),
            SafetyProperty::InteractionSeverityLimit {
                drug_a,
                drug_b,
                max_severity,
            } => self.check_interaction_severity(state, drug_a, drug_b, *max_severity),
            SafetyProperty::Conjunction(props) => {
                self.check_conjunction(state, props, property)
            }
            SafetyProperty::Disjunction(props) => {
                self.check_disjunction(state, props, property)
            }
        }
    }

    // ── Conjunction / Disjunction helpers ───────────────────────────────

    fn check_conjunction(
        &self,
        state: &ProductDomain,
        props: &[SafetyProperty],
        outer: &SafetyProperty,
    ) -> SafetyEvidence {
        let evidences: Vec<SafetyEvidence> =
            props.iter().map(|p| self.check_safety(state, p)).collect();
        let mut combined = SafetyVerdict::Safe;
        for ev in &evidences {
            combined = SafetyVerdict::combine_and(&combined, &ev.verdict);
        }
        let mut result = SafetyEvidence::new(combined, outer.description());
        for ev in evidences {
            for pair in ev.concentration_evidence {
                result.concentration_evidence.push(pair);
            }
            for triple in ev.enzyme_evidence {
                result.enzyme_evidence.push(triple);
            }
            for d in ev.details {
                result.details.push(d);
            }
        }
        result
    }

    fn check_disjunction(
        &self,
        state: &ProductDomain,
        props: &[SafetyProperty],
        outer: &SafetyProperty,
    ) -> SafetyEvidence {
        let evidences: Vec<SafetyEvidence> =
            props.iter().map(|p| self.check_safety(state, p)).collect();
        let mut combined = SafetyVerdict::Unsafe;
        for ev in &evidences {
            combined = SafetyVerdict::combine_or(&combined, &ev.verdict);
        }
        let mut result = SafetyEvidence::new(combined, outer.description());
        for ev in evidences {
            for pair in ev.concentration_evidence {
                result.concentration_evidence.push(pair);
            }
            for triple in ev.enzyme_evidence {
                result.enzyme_evidence.push(triple);
            }
            for d in ev.details {
                result.details.push(d);
            }
        }
        result
    }

    // ── Concentration bound ────────────────────────────────────────────

    pub fn check_concentration_bound(
        &self,
        state: &ProductDomain,
        drug: &DrugId,
        lo: f64,
        hi: f64,
    ) -> SafetyEvidence {
        let interval = state.concentration(drug);
        let desc = format!(
            "Concentration of {} in [{:.4}, {:.4}]",
            drug.as_str(),
            lo,
            hi,
        );

        if interval.is_bottom() {
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown("drug concentration is bottom (absent)".into()),
                desc,
            );
            ev.add_detail(format!(
                "Drug {} has no concentration data in current state",
                drug.as_str()
            ));
            return ev;
        }

        let bound = ConcentrationInterval::new(lo, hi);

        if bound.contains_interval(&interval) {
            // [interval.lo, interval.hi] ⊆ [lo, hi] → definitely safe
            let mut ev = SafetyEvidence::new(SafetyVerdict::Safe, desc)
                .with_concentration(drug.clone(), interval);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] ⊆ [{:.4}, {:.4}]",
                interval.lo, interval.hi, lo, hi,
            ));
            ev
        } else if !interval.overlaps(&bound) {
            // No overlap → definitely unsafe
            let mut ev = SafetyEvidence::new(SafetyVerdict::Unsafe, desc)
                .with_concentration(drug.clone(), interval);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] ∩ [{:.4}, {:.4}] = ∅",
                interval.lo, interval.hi, lo, hi,
            ));
            ev
        } else {
            // Partial overlap → unknown
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown("partial overlap with bound".into()),
                desc,
            )
            .with_concentration(drug.clone(), interval);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] partially overlaps [{:.4}, {:.4}]",
                interval.lo, interval.hi, lo, hi,
            ));
            ev
        }
    }

    // ── Therapeutic window ─────────────────────────────────────────────

    pub fn check_therapeutic_window(
        &self,
        state: &ProductDomain,
        drug: &DrugId,
        window: &TherapeuticWindow,
    ) -> SafetyEvidence {
        let interval = state.concentration(drug);
        let desc = format!(
            "Concentration of {} in therapeutic window [{:.4}, {:.4}]",
            drug.as_str(),
            window.min_concentration,
            window.max_concentration,
        );

        if interval.is_bottom() {
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown("drug concentration is bottom (absent)".into()),
                desc,
            );
            ev.add_detail(format!(
                "Drug {} has no concentration data",
                drug.as_str()
            ));
            return ev;
        }

        let tw_interval = window.to_interval();

        if tw_interval.contains_interval(&interval) {
            let mut ev = SafetyEvidence::new(SafetyVerdict::Safe, desc)
                .with_concentration(drug.clone(), interval);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] ⊆ [{:.4}, {:.4}]",
                interval.lo,
                interval.hi,
                window.min_concentration,
                window.max_concentration,
            ));
            ev
        } else if !interval.overlaps(&tw_interval) {
            let mut ev = SafetyEvidence::new(SafetyVerdict::Unsafe, desc)
                .with_concentration(drug.clone(), interval);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] ∩ [{:.4}, {:.4}] = ∅",
                interval.lo,
                interval.hi,
                window.min_concentration,
                window.max_concentration,
            ));
            ev
        } else {
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown(
                    "partial overlap with therapeutic window".into(),
                ),
                desc,
            )
            .with_concentration(drug.clone(), interval);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] partially overlaps [{:.4}, {:.4}]",
                interval.lo,
                interval.hi,
                window.min_concentration,
                window.max_concentration,
            ));
            ev
        }
    }

    // ── Enzyme activity bound ──────────────────────────────────────────

    pub fn check_enzyme_bound(
        &self,
        state: &ProductDomain,
        enzyme: &CypEnzyme,
        min_act: f64,
        max_act: f64,
    ) -> SafetyEvidence {
        let activity = state.enzyme_activity(enzyme);
        let desc = format!(
            "Activity of {:?} in [{:.4}, {:.4}]",
            enzyme, min_act, max_act,
        );

        if activity.is_bottom() {
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown("enzyme activity is bottom".into()),
                desc,
            );
            ev.add_detail(format!("Enzyme {:?} has bottom activity", enzyme));
            return ev;
        }

        let fully_within = activity.lo >= min_act && activity.hi <= max_act;
        let no_overlap = activity.hi < min_act || activity.lo > max_act;

        if fully_within {
            let mut ev = SafetyEvidence::new(SafetyVerdict::Safe, desc)
                .with_enzyme(*enzyme, activity.lo, activity.hi);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] ⊆ [{:.4}, {:.4}]",
                activity.lo, activity.hi, min_act, max_act,
            ));
            ev
        } else if no_overlap {
            let mut ev = SafetyEvidence::new(SafetyVerdict::Unsafe, desc)
                .with_enzyme(*enzyme, activity.lo, activity.hi);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] ∩ [{:.4}, {:.4}] = ∅",
                activity.lo, activity.hi, min_act, max_act,
            ));
            ev
        } else {
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown("partial overlap with enzyme bound".into()),
                desc,
            )
            .with_enzyme(*enzyme, activity.lo, activity.hi);
            ev.add_detail(format!(
                "[{:.4}, {:.4}] partially overlaps [{:.4}, {:.4}]",
                activity.lo, activity.hi, min_act, max_act,
            ));
            ev
        }
    }

    // ── Interaction severity ───────────────────────────────────────────

    /// Compute the AUC-ratio–based severity of an interaction between two drugs
    /// and compare it against the allowed maximum severity.
    pub fn check_interaction_severity(
        &self,
        state: &ProductDomain,
        drug_a: &DrugId,
        drug_b: &DrugId,
        max_severity: Severity,
    ) -> SafetyEvidence {
        let desc = format!(
            "Interaction {} ↔ {} ≤ {:?}",
            drug_a.as_str(),
            drug_b.as_str(),
            max_severity,
        );

        let conc_a = state.concentration(drug_a);
        let conc_b = state.concentration(drug_b);

        if conc_a.is_bottom() || conc_b.is_bottom() {
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown(
                    "one or both drug concentrations are absent".into(),
                ),
                desc,
            );
            ev.add_detail(
                "Cannot assess interaction without concentration data".into(),
            );
            return ev;
        }

        let config_a = self.find_config(drug_a);
        let config_b = self.find_config(drug_b);

        if config_a.is_none() && config_b.is_none() {
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown("no drug configs available".into()),
                desc,
            );
            ev.add_detail(
                "Cannot assess interaction without drug configurations".into(),
            );
            return ev;
        }

        let mut worst_auc_lo = 1.0_f64;
        let mut worst_auc_hi = 1.0_f64;
        let mut found_interaction = false;
        let mut evidence = SafetyEvidence::new(SafetyVerdict::Safe, desc.clone());

        // Direction: A inhibits B's metabolism
        if let (Some(ca), Some(cb)) = (config_a, config_b) {
            for inh in &ca.inhibition_effects {
                let victim_fraction: f64 = cb
                    .metabolism_routes
                    .iter()
                    .filter(|r| r.enzyme == inh.enzyme)
                    .map(|r| r.fraction_metabolized)
                    .sum();
                if victim_fraction > 0.0 {
                    found_interaction = true;
                    let ratio_at_lo = inh.auc_ratio(conc_a.lo);
                    let ratio_at_hi = inh.auc_ratio(conc_a.hi);
                    let r_lo = ratio_at_lo.min(ratio_at_hi);
                    let r_hi = ratio_at_lo.max(ratio_at_hi);
                    let scaled_lo = 1.0 + (r_lo - 1.0) * victim_fraction;
                    let scaled_hi = 1.0 + (r_hi - 1.0) * victim_fraction;
                    worst_auc_lo = worst_auc_lo.max(scaled_lo);
                    worst_auc_hi = worst_auc_hi.max(scaled_hi);
                    evidence = evidence.with_enzyme(inh.enzyme, r_lo, r_hi);
                    evidence.add_detail(format!(
                        "{} inhibits {:?} (fm={:.2}): AUC ratio [{:.3}, {:.3}]",
                        drug_a.as_str(),
                        inh.enzyme,
                        victim_fraction,
                        scaled_lo,
                        scaled_hi,
                    ));
                }
            }
        }

        // Direction: B inhibits A's metabolism
        if let (Some(cb), Some(ca)) = (config_b, config_a) {
            for inh in &cb.inhibition_effects {
                let victim_fraction: f64 = ca
                    .metabolism_routes
                    .iter()
                    .filter(|r| r.enzyme == inh.enzyme)
                    .map(|r| r.fraction_metabolized)
                    .sum();
                if victim_fraction > 0.0 {
                    found_interaction = true;
                    let ratio_at_lo = inh.auc_ratio(conc_b.lo);
                    let ratio_at_hi = inh.auc_ratio(conc_b.hi);
                    let r_lo = ratio_at_lo.min(ratio_at_hi);
                    let r_hi = ratio_at_lo.max(ratio_at_hi);
                    let scaled_lo = 1.0 + (r_lo - 1.0) * victim_fraction;
                    let scaled_hi = 1.0 + (r_hi - 1.0) * victim_fraction;
                    worst_auc_lo = worst_auc_lo.max(scaled_lo);
                    worst_auc_hi = worst_auc_hi.max(scaled_hi);
                    evidence = evidence.with_enzyme(inh.enzyme, r_lo, r_hi);
                    evidence.add_detail(format!(
                        "{} inhibits {:?} (fm={:.2}): AUC ratio [{:.3}, {:.3}]",
                        drug_b.as_str(),
                        inh.enzyme,
                        victim_fraction,
                        scaled_lo,
                        scaled_hi,
                    ));
                }
            }
        }

        if !found_interaction {
            evidence.verdict = SafetyVerdict::Safe;
            evidence.add_detail(
                "No shared metabolic pathways found; no interaction".into(),
            );
            return evidence;
        }

        let sev_lo = Self::severity_from_auc_ratio(worst_auc_lo);
        let sev_hi = Self::severity_from_auc_ratio(worst_auc_hi);

        evidence.add_detail(format!(
            "Worst-case AUC ratio range: [{:.3}, {:.3}]",
            worst_auc_lo, worst_auc_hi,
        ));
        evidence.add_detail(format!(
            "Severity range: {:?} to {:?}",
            sev_lo, sev_hi,
        ));
        evidence = evidence
            .with_concentration(drug_a.clone(), conc_a)
            .with_concentration(drug_b.clone(), conc_b);

        if sev_hi <= max_severity {
            evidence.verdict = SafetyVerdict::Safe;
        } else if sev_lo > max_severity {
            evidence.verdict = SafetyVerdict::Unsafe;
        } else {
            evidence.verdict = SafetyVerdict::Unknown(format!(
                "severity range [{:?}, {:?}] straddles limit {:?}",
                sev_lo, sev_hi, max_severity,
            ));
        }

        evidence
    }

    // ── Comprehensive pair classification ──────────────────────────────

    /// Classify the interaction between two drugs across all shared enzymes.
    pub fn classify_drug_pair(
        &self,
        state: &ProductDomain,
        drug_a: &DrugId,
        drug_b: &DrugId,
    ) -> SafetyEvidence {
        let desc = format!(
            "Comprehensive interaction: {} ↔ {}",
            drug_a.as_str(),
            drug_b.as_str(),
        );

        let conc_a = state.concentration(drug_a);
        let conc_b = state.concentration(drug_b);

        if conc_a.is_bottom() || conc_b.is_bottom() {
            let mut ev = SafetyEvidence::new(
                SafetyVerdict::Unknown(
                    "one or both drug concentrations are absent".into(),
                ),
                desc,
            );
            ev.add_detail(
                "Cannot classify pair without concentration data".into(),
            );
            return ev;
        }

        let config_a = self.find_config(drug_a);
        let config_b = self.find_config(drug_b);

        let mut combined_auc_lo = 1.0_f64;
        let mut combined_auc_hi = 1.0_f64;
        let mut shared_enzymes: Vec<CypEnzyme> = Vec::new();
        let mut evidence = SafetyEvidence::new(SafetyVerdict::Safe, desc)
            .with_concentration(drug_a.clone(), conc_a)
            .with_concentration(drug_b.clone(), conc_b);

        // A inhibits B's metabolism
        if let (Some(ca), Some(cb)) = (config_a, config_b) {
            for inh in &ca.inhibition_effects {
                let victim_fraction: f64 = cb
                    .metabolism_routes
                    .iter()
                    .filter(|r| r.enzyme == inh.enzyme)
                    .map(|r| r.fraction_metabolized)
                    .sum();
                if victim_fraction > 0.0 {
                    if !shared_enzymes.contains(&inh.enzyme) {
                        shared_enzymes.push(inh.enzyme);
                    }
                    let ratio_at_lo = inh.auc_ratio(conc_a.lo);
                    let ratio_at_hi = inh.auc_ratio(conc_a.hi);
                    let r_lo = ratio_at_lo.min(ratio_at_hi);
                    let r_hi = ratio_at_lo.max(ratio_at_hi);
                    let scaled_lo = 1.0 + (r_lo - 1.0) * victim_fraction;
                    let scaled_hi = 1.0 + (r_hi - 1.0) * victim_fraction;
                    combined_auc_lo = combined_auc_lo.max(scaled_lo);
                    combined_auc_hi = combined_auc_hi.max(scaled_hi);
                    evidence = evidence.with_enzyme(inh.enzyme, r_lo, r_hi);
                    evidence.add_detail(format!(
                        "{} → {:?} → {}: fm={:.2}, AUC ratio [{:.3}, {:.3}]",
                        drug_a.as_str(),
                        inh.enzyme,
                        drug_b.as_str(),
                        victim_fraction,
                        scaled_lo,
                        scaled_hi,
                    ));
                }
            }
        }

        // B inhibits A's metabolism
        if let (Some(cb), Some(ca)) = (config_b, config_a) {
            for inh in &cb.inhibition_effects {
                let victim_fraction: f64 = ca
                    .metabolism_routes
                    .iter()
                    .filter(|r| r.enzyme == inh.enzyme)
                    .map(|r| r.fraction_metabolized)
                    .sum();
                if victim_fraction > 0.0 {
                    if !shared_enzymes.contains(&inh.enzyme) {
                        shared_enzymes.push(inh.enzyme);
                    }
                    let ratio_at_lo = inh.auc_ratio(conc_b.lo);
                    let ratio_at_hi = inh.auc_ratio(conc_b.hi);
                    let r_lo = ratio_at_lo.min(ratio_at_hi);
                    let r_hi = ratio_at_lo.max(ratio_at_hi);
                    let scaled_lo = 1.0 + (r_lo - 1.0) * victim_fraction;
                    let scaled_hi = 1.0 + (r_hi - 1.0) * victim_fraction;
                    combined_auc_lo = combined_auc_lo.max(scaled_lo);
                    combined_auc_hi = combined_auc_hi.max(scaled_hi);
                    evidence = evidence.with_enzyme(inh.enzyme, r_lo, r_hi);
                    evidence.add_detail(format!(
                        "{} → {:?} → {}: fm={:.2}, AUC ratio [{:.3}, {:.3}]",
                        drug_b.as_str(),
                        inh.enzyme,
                        drug_a.as_str(),
                        victim_fraction,
                        scaled_lo,
                        scaled_hi,
                    ));
                }
            }
        }

        if shared_enzymes.is_empty() {
            evidence.verdict = SafetyVerdict::Safe;
            evidence.add_detail(
                "No shared metabolic pathways; no interaction expected".into(),
            );
            return evidence;
        }

        let sev_lo = Self::severity_from_auc_ratio(combined_auc_lo);
        let sev_hi = Self::severity_from_auc_ratio(combined_auc_hi);

        evidence.add_detail(format!(
            "Combined AUC ratio: [{:.3}, {:.3}]",
            combined_auc_lo, combined_auc_hi,
        ));
        evidence.add_detail(format!(
            "Estimated severity: {:?} to {:?}",
            sev_lo, sev_hi,
        ));
        evidence.add_detail(format!("Shared enzymes: {:?}", shared_enzymes));

        if sev_hi >= Severity::Major {
            evidence.verdict = SafetyVerdict::Unsafe;
        } else if sev_hi <= Severity::Minor {
            evidence.verdict = SafetyVerdict::Safe;
        } else {
            evidence.verdict = SafetyVerdict::Unknown(format!(
                "moderate interaction detected: {:?} to {:?}",
                sev_lo, sev_hi,
            ));
        }

        evidence
    }

    // ── Batch check ────────────────────────────────────────────────────

    pub fn check_all_properties(
        &self,
        state: &ProductDomain,
        properties: &[SafetyProperty],
    ) -> Vec<SafetyEvidence> {
        properties
            .iter()
            .map(|p| self.check_safety(state, p))
            .collect()
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        DosingSchedule, InhibitionEffect, MetabolismRoute, PkParameters,
    };

    // ── Helpers ────────────────────────────────────────────────────────

    fn drug(name: &str) -> DrugId {
        DrugId::new(name)
    }

    fn simple_config(name: &str) -> DrugConfig {
        DrugConfig::new(
            DrugId::new(name),
            DosingSchedule::new(100.0, 12.0),
            PkParameters::new(5.0, 50.0),
        )
    }

    fn state_with_conc(pairs: &[(&str, f64, f64)]) -> ProductDomain {
        let mut s = ProductDomain::initial();
        for &(name, lo, hi) in pairs {
            s.set_concentration(DrugId::new(name), ConcentrationInterval::new(lo, hi));
        }
        s
    }

    // ── 1. Safe concentration bound ────────────────────────────────────

    #[test]
    fn test_concentration_bound_safe() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("aspirin", 2.0, 4.0)]);
        let ev = checker.check_concentration_bound(&state, &drug("aspirin"), 1.0, 5.0);
        assert!(ev.verdict.is_safe());
        assert!(!ev.concentration_evidence.is_empty());
    }

    // ── 2. Unsafe concentration bound ──────────────────────────────────

    #[test]
    fn test_concentration_bound_unsafe() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("aspirin", 10.0, 20.0)]);
        let ev = checker.check_concentration_bound(&state, &drug("aspirin"), 1.0, 5.0);
        assert!(ev.verdict.is_unsafe());
    }

    // ── 3. Unknown concentration bound (partial overlap) ───────────────

    #[test]
    fn test_concentration_bound_unknown() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("aspirin", 3.0, 8.0)]);
        let ev = checker.check_concentration_bound(&state, &drug("aspirin"), 1.0, 5.0);
        assert!(ev.verdict.is_unknown());
    }

    // ── 4. Therapeutic window – within ─────────────────────────────────

    #[test]
    fn test_therapeutic_window_within() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("warfarin", 1.5, 2.5)]);
        let window = TherapeuticWindow::new(1.0, 3.0);
        let ev = checker.check_therapeutic_window(&state, &drug("warfarin"), &window);
        assert!(ev.verdict.is_safe());
    }

    // ── 5. Therapeutic window – outside ────────────────────────────────

    #[test]
    fn test_therapeutic_window_outside() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("warfarin", 5.0, 8.0)]);
        let window = TherapeuticWindow::new(1.0, 3.0);
        let ev = checker.check_therapeutic_window(&state, &drug("warfarin"), &window);
        assert!(ev.verdict.is_unsafe());
    }

    // ── 6. Enzyme activity within bounds ───────────────────────────────

    #[test]
    fn test_enzyme_activity_within_bounds() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let mut state = ProductDomain::initial();
        state.set_enzyme_activity(
            CypEnzyme::CYP3A4,
            EnzymeActivityAbstractInterval::new(0.8, 1.2),
        );
        let ev = checker.check_enzyme_bound(&state, &CypEnzyme::CYP3A4, 0.5, 1.5);
        assert!(ev.verdict.is_safe());
    }

    // ── 7. Enzyme activity outside bounds ──────────────────────────────

    #[test]
    fn test_enzyme_activity_outside_bounds() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let mut state = ProductDomain::initial();
        state.set_enzyme_activity(
            CypEnzyme::CYP2D6,
            EnzymeActivityAbstractInterval::new(0.1, 0.2),
        );
        let ev = checker.check_enzyme_bound(&state, &CypEnzyme::CYP2D6, 0.5, 1.5);
        assert!(ev.verdict.is_unsafe());
    }

    // ── 8. Interaction severity classification ─────────────────────────

    #[test]
    fn test_interaction_severity_safe() {
        // Drug A inhibits CYP3A4, Drug B is metabolized by CYP3A4.
        // With low inhibitor concentration the AUC ratio stays small.
        let config_a = simple_config("inhibitor")
            .with_inhibition(InhibitionEffect::new(
                CypEnzyme::CYP3A4,
                InhibitionType::Competitive,
                100.0, // large Ki → weak inhibition
            ));
        let config_b = simple_config("victim")
            .with_metabolism(MetabolismRoute::new(CypEnzyme::CYP3A4, 0.8));
        let checker = AbstractSafetyChecker::new(vec![config_a, config_b]);

        // Low inhibitor concentration → minor interaction
        let state = state_with_conc(&[("inhibitor", 0.5, 1.0), ("victim", 2.0, 4.0)]);
        let ev = checker.check_interaction_severity(
            &state,
            &drug("inhibitor"),
            &drug("victim"),
            Severity::Moderate,
        );
        assert!(
            ev.verdict.is_safe(),
            "Expected safe, got {:?}: {:?}",
            ev.verdict,
            ev.details
        );
    }

    // ── 9. classify_drug_pair with inhibition ──────────────────────────

    #[test]
    fn test_classify_drug_pair_with_inhibition() {
        let config_a = simple_config("ketoconazole")
            .with_inhibition(InhibitionEffect::new(
                CypEnzyme::CYP3A4,
                InhibitionType::Competitive,
                0.015, // very potent → Ki in µM
            ));
        let config_b = simple_config("midazolam")
            .with_metabolism(MetabolismRoute::new(CypEnzyme::CYP3A4, 0.95));
        let checker = AbstractSafetyChecker::new(vec![config_a, config_b]);

        // High inhibitor concentration with potent Ki → large AUC ratio → Unsafe
        let state =
            state_with_conc(&[("ketoconazole", 5.0, 10.0), ("midazolam", 0.5, 1.0)]);
        let ev = checker.classify_drug_pair(
            &state,
            &drug("ketoconazole"),
            &drug("midazolam"),
        );
        assert!(
            ev.verdict.is_unsafe(),
            "Expected unsafe for potent inhibition, got {:?}: {:?}",
            ev.verdict,
            ev.details,
        );
        assert!(!ev.enzyme_evidence.is_empty());
    }

    // ── 10. Conjunction of properties ──────────────────────────────────

    #[test]
    fn test_conjunction_all_safe() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("a", 2.0, 3.0), ("b", 5.0, 6.0)]);

        let prop = SafetyProperty::Conjunction(vec![
            SafetyProperty::ConcentrationBound {
                drug: drug("a"),
                lo: 1.0,
                hi: 4.0,
            },
            SafetyProperty::ConcentrationBound {
                drug: drug("b"),
                lo: 4.0,
                hi: 7.0,
            },
        ]);
        let ev = checker.check_safety(&state, &prop);
        assert!(ev.verdict.is_safe());
    }

    #[test]
    fn test_conjunction_one_unsafe() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("a", 2.0, 3.0), ("b", 50.0, 60.0)]);

        let prop = SafetyProperty::Conjunction(vec![
            SafetyProperty::ConcentrationBound {
                drug: drug("a"),
                lo: 1.0,
                hi: 4.0,
            },
            SafetyProperty::ConcentrationBound {
                drug: drug("b"),
                lo: 4.0,
                hi: 7.0,
            },
        ]);
        let ev = checker.check_safety(&state, &prop);
        assert!(ev.verdict.is_unsafe());
    }

    // ── 11. Disjunction of properties ──────────────────────────────────

    #[test]
    fn test_disjunction_one_safe() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("a", 2.0, 3.0), ("b", 50.0, 60.0)]);

        let prop = SafetyProperty::Disjunction(vec![
            SafetyProperty::ConcentrationBound {
                drug: drug("a"),
                lo: 1.0,
                hi: 4.0,
            },
            SafetyProperty::ConcentrationBound {
                drug: drug("b"),
                lo: 4.0,
                hi: 7.0,
            },
        ]);
        let ev = checker.check_safety(&state, &prop);
        assert!(ev.verdict.is_safe());
    }

    #[test]
    fn test_disjunction_all_unsafe() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("a", 20.0, 30.0), ("b", 50.0, 60.0)]);

        let prop = SafetyProperty::Disjunction(vec![
            SafetyProperty::ConcentrationBound {
                drug: drug("a"),
                lo: 1.0,
                hi: 4.0,
            },
            SafetyProperty::ConcentrationBound {
                drug: drug("b"),
                lo: 4.0,
                hi: 7.0,
            },
        ]);
        let ev = checker.check_safety(&state, &prop);
        assert!(ev.verdict.is_unsafe());
    }

    // ── 12. check_all_properties ───────────────────────────────────────

    #[test]
    fn test_check_all_properties_count() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("x", 1.0, 2.0)]);
        let props = vec![
            SafetyProperty::ConcentrationBound {
                drug: drug("x"),
                lo: 0.0,
                hi: 3.0,
            },
            SafetyProperty::ConcentrationBound {
                drug: drug("x"),
                lo: 5.0,
                hi: 10.0,
            },
            SafetyProperty::EnzymeActivityBound {
                enzyme: CypEnzyme::CYP3A4,
                min_activity: 0.5,
                max_activity: 1.5,
            },
        ];
        let results = checker.check_all_properties(&state, &props);
        assert_eq!(results.len(), 3);
        assert!(results[0].verdict.is_safe());
        assert!(results[1].verdict.is_unsafe());
    }

    // ── 13. Edge case: bottom state ────────────────────────────────────

    #[test]
    fn test_bottom_state_yields_unknown() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = ProductDomain::bottom();
        let ev = checker.check_concentration_bound(&state, &drug("any"), 0.0, 10.0);
        assert!(ev.verdict.is_unknown());
    }

    // ── 14. Edge case: drug not in state ───────────────────────────────

    #[test]
    fn test_drug_not_in_state() {
        let checker = AbstractSafetyChecker::new(vec![]);
        let state = state_with_conc(&[("present", 1.0, 2.0)]);
        let ev =
            checker.check_concentration_bound(&state, &drug("absent"), 0.0, 10.0);
        assert!(ev.verdict.is_unknown());
        assert!(ev
            .details
            .iter()
            .any(|d| d.contains("no concentration data")));
    }

    // ── Extra: SafetyVerdict combinators ───────────────────────────────

    #[test]
    fn test_verdict_combine_and() {
        assert_eq!(
            SafetyVerdict::combine_and(&SafetyVerdict::Safe, &SafetyVerdict::Safe),
            SafetyVerdict::Safe,
        );
        assert_eq!(
            SafetyVerdict::combine_and(
                &SafetyVerdict::Safe,
                &SafetyVerdict::Unsafe
            ),
            SafetyVerdict::Unsafe,
        );
        let v = SafetyVerdict::combine_and(
            &SafetyVerdict::Safe,
            &SafetyVerdict::Unknown("x".into()),
        );
        assert!(v.is_unknown());
    }

    #[test]
    fn test_verdict_combine_or() {
        assert_eq!(
            SafetyVerdict::combine_or(&SafetyVerdict::Safe, &SafetyVerdict::Unsafe),
            SafetyVerdict::Safe,
        );
        assert_eq!(
            SafetyVerdict::combine_or(
                &SafetyVerdict::Unsafe,
                &SafetyVerdict::Unsafe
            ),
            SafetyVerdict::Unsafe,
        );
        let v = SafetyVerdict::combine_or(
            &SafetyVerdict::Unsafe,
            &SafetyVerdict::Unknown("y".into()),
        );
        assert!(v.is_unknown());
    }

    // ── Extra: SafetyProperty description / referenced_drugs ───────────

    #[test]
    fn test_property_referenced_drugs() {
        let prop = SafetyProperty::Conjunction(vec![
            SafetyProperty::ConcentrationBound {
                drug: drug("a"),
                lo: 0.0,
                hi: 1.0,
            },
            SafetyProperty::InteractionSeverityLimit {
                drug_a: drug("a"),
                drug_b: drug("b"),
                max_severity: Severity::Major,
            },
            SafetyProperty::EnzymeActivityBound {
                enzyme: CypEnzyme::CYP2D6,
                min_activity: 0.5,
                max_activity: 1.5,
            },
        ]);
        let drugs = prop.referenced_drugs();
        assert_eq!(drugs.len(), 2);
        assert!(drugs.contains(&drug("a")));
        assert!(drugs.contains(&drug("b")));
    }

    // ── Extra: no shared enzyme → safe interaction ─────────────────────

    #[test]
    fn test_no_shared_enzyme_safe() {
        let config_a = simple_config("drug_a")
            .with_inhibition(InhibitionEffect::new(
                CypEnzyme::CYP2D6,
                InhibitionType::Competitive,
                1.0,
            ));
        let config_b = simple_config("drug_b")
            .with_metabolism(MetabolismRoute::new(CypEnzyme::CYP3A4, 0.9));
        let checker = AbstractSafetyChecker::new(vec![config_a, config_b]);

        let state = state_with_conc(&[("drug_a", 1.0, 2.0), ("drug_b", 3.0, 4.0)]);
        let ev = checker.check_interaction_severity(
            &state,
            &drug("drug_a"),
            &drug("drug_b"),
            Severity::Minor,
        );
        assert!(ev.verdict.is_safe());
    }
}

//! Safety property and verdict types for abstract interpretation.
//!
//! Defines the properties to be checked, verdicts from checking, and
//! evidence structures that explain why a particular verdict was reached.

use serde::{Deserialize, Serialize};
use std::fmt;
use guardpharma_types::{DrugId, CypEnzyme, Severity, ConcentrationInterval};

// ---------------------------------------------------------------------------
// Safety Property
// ---------------------------------------------------------------------------

/// A safety property to verify over abstract states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyProperty {
    pub id: String,
    pub name: String,
    pub kind: SafetyPropertyKind,
}

impl SafetyProperty {
    pub fn new(id: impl Into<String>, name: impl Into<String>, kind: SafetyPropertyKind) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            kind,
        }
    }

    pub fn concentration_bound(drug: DrugId, lo: f64, hi: f64) -> Self {
        Self::new(
            format!("conc_bound_{}", drug),
            format!("Concentration bound for {}", drug),
            SafetyPropertyKind::ConcentrationBound { drug, lo, hi },
        )
    }

    pub fn interaction_limit(drug_a: DrugId, drug_b: DrugId, max_severity: Severity) -> Self {
        Self::new(
            format!("interaction_{}_{}", drug_a, drug_b),
            format!("Interaction limit {} vs {}", drug_a, drug_b),
            SafetyPropertyKind::InteractionLimit { drug_a, drug_b, max_severity },
        )
    }

    pub fn enzyme_activity_bound(enzyme: CypEnzyme, lo: f64, hi: f64) -> Self {
        Self::new(
            format!("enzyme_bound_{}", enzyme),
            format!("Enzyme activity bound for {}", enzyme),
            SafetyPropertyKind::EnzymeActivityBound { enzyme, lo, hi },
        )
    }

    pub fn therapeutic_window(drug: DrugId, lower: f64, upper: f64) -> Self {
        Self::new(
            format!("tw_{}", drug),
            format!("Therapeutic window for {}", drug),
            SafetyPropertyKind::TherapeuticWindow { drug, lower, upper },
        )
    }

    pub fn conjunction(properties: Vec<SafetyProperty>) -> Self {
        Self::new(
            "conjunction".to_string(),
            "Conjunction of properties".to_string(),
            SafetyPropertyKind::And(properties),
        )
    }

    pub fn disjunction(properties: Vec<SafetyProperty>) -> Self {
        Self::new(
            "disjunction".to_string(),
            "Disjunction of properties".to_string(),
            SafetyPropertyKind::Or(properties),
        )
    }

    /// Collect all drug IDs referenced in this property.
    pub fn referenced_drugs(&self) -> Vec<DrugId> {
        match &self.kind {
            SafetyPropertyKind::ConcentrationBound { drug, .. } => vec![drug.clone()],
            SafetyPropertyKind::TherapeuticWindow { drug, .. } => vec![drug.clone()],
            SafetyPropertyKind::InteractionLimit { drug_a, drug_b, .. } => {
                vec![drug_a.clone(), drug_b.clone()]
            }
            SafetyPropertyKind::EnzymeActivityBound { .. } => Vec::new(),
            SafetyPropertyKind::TemporalBound { drug, .. } => vec![drug.clone()],
            SafetyPropertyKind::And(props) | SafetyPropertyKind::Or(props) => {
                props.iter().flat_map(|p| p.referenced_drugs()).collect()
            }
        }
    }

    /// Collect all enzymes referenced in this property.
    pub fn referenced_enzymes(&self) -> Vec<CypEnzyme> {
        match &self.kind {
            SafetyPropertyKind::EnzymeActivityBound { enzyme, .. } => vec![*enzyme],
            SafetyPropertyKind::And(props) | SafetyPropertyKind::Or(props) => {
                props.iter().flat_map(|p| p.referenced_enzymes()).collect()
            }
            _ => Vec::new(),
        }
    }
}

impl fmt::Display for SafetyProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.id, self.name)
    }
}

/// The kind of safety property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyPropertyKind {
    /// Drug concentration must stay within [lo, hi].
    ConcentrationBound {
        drug: DrugId,
        lo: f64,
        hi: f64,
    },
    /// Drug concentration must stay within therapeutic window.
    TherapeuticWindow {
        drug: DrugId,
        lower: f64,
        upper: f64,
    },
    /// Drug interaction severity must not exceed max_severity.
    InteractionLimit {
        drug_a: DrugId,
        drug_b: DrugId,
        max_severity: Severity,
    },
    /// Enzyme activity must stay within [lo, hi].
    EnzymeActivityBound {
        enzyme: CypEnzyme,
        lo: f64,
        hi: f64,
    },
    /// Temporal bound: concentration must reach target within time_hours.
    TemporalBound {
        drug: DrugId,
        target_concentration: f64,
        time_hours: f64,
    },
    /// Conjunction of multiple properties (all must hold).
    And(Vec<SafetyProperty>),
    /// Disjunction of multiple properties (at least one must hold).
    Or(Vec<SafetyProperty>),
}

// ---------------------------------------------------------------------------
// Safety Verdict
// ---------------------------------------------------------------------------

/// The result of checking a safety property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyVerdict {
    /// The property is definitely satisfied by all concrete states in the
    /// abstract state.
    Safe {
        evidence: SafetyEvidence,
    },
    /// The property is definitely violated by all concrete states in the
    /// abstract state.
    Unsafe {
        evidence: SafetyEvidence,
    },
    /// The abstract state is too imprecise to determine — some concrete
    /// states may satisfy the property while others violate it.
    Unknown {
        reason: String,
        evidence: SafetyEvidence,
    },
}

impl SafetyVerdict {
    pub fn safe(evidence: SafetyEvidence) -> Self {
        SafetyVerdict::Safe { evidence }
    }

    pub fn unsafe_verdict(evidence: SafetyEvidence) -> Self {
        SafetyVerdict::Unsafe { evidence }
    }

    pub fn unknown(reason: impl Into<String>, evidence: SafetyEvidence) -> Self {
        SafetyVerdict::Unknown {
            reason: reason.into(),
            evidence,
        }
    }

    pub fn is_safe(&self) -> bool {
        matches!(self, SafetyVerdict::Safe { .. })
    }

    pub fn is_unsafe(&self) -> bool {
        matches!(self, SafetyVerdict::Unsafe { .. })
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, SafetyVerdict::Unknown { .. })
    }

    pub fn needs_tier2(&self) -> bool {
        !self.is_safe()
    }

    pub fn evidence(&self) -> &SafetyEvidence {
        match self {
            SafetyVerdict::Safe { evidence }
            | SafetyVerdict::Unsafe { evidence }
            | SafetyVerdict::Unknown { evidence, .. } => evidence,
        }
    }

    /// Combine two verdicts conservatively: safe only if both safe.
    pub fn combine_and(a: &SafetyVerdict, b: &SafetyVerdict) -> SafetyVerdict {
        match (a, b) {
            (SafetyVerdict::Safe { evidence: ea }, SafetyVerdict::Safe { evidence: eb }) => {
                SafetyVerdict::Safe {
                    evidence: SafetyEvidence::combined(ea, eb),
                }
            }
            (SafetyVerdict::Unsafe { evidence }, _) | (_, SafetyVerdict::Unsafe { evidence }) => {
                SafetyVerdict::Unsafe { evidence: evidence.clone() }
            }
            _ => {
                let ev = SafetyEvidence::combined(a.evidence(), b.evidence());
                SafetyVerdict::Unknown {
                    reason: "One or more sub-properties inconclusive".into(),
                    evidence: ev,
                }
            }
        }
    }

    /// Combine two verdicts disjunctively: safe if either safe.
    pub fn combine_or(a: &SafetyVerdict, b: &SafetyVerdict) -> SafetyVerdict {
        match (a, b) {
            (SafetyVerdict::Safe { evidence }, _) | (_, SafetyVerdict::Safe { evidence }) => {
                SafetyVerdict::Safe { evidence: evidence.clone() }
            }
            (SafetyVerdict::Unsafe { evidence: ea }, SafetyVerdict::Unsafe { evidence: eb }) => {
                SafetyVerdict::Unsafe {
                    evidence: SafetyEvidence::combined(ea, eb),
                }
            }
            _ => {
                let ev = SafetyEvidence::combined(a.evidence(), b.evidence());
                SafetyVerdict::Unknown {
                    reason: "Disjunction inconclusive".into(),
                    evidence: ev,
                }
            }
        }
    }
}

impl fmt::Display for SafetyVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafetyVerdict::Safe { .. } => write!(f, "SAFE"),
            SafetyVerdict::Unsafe { .. } => write!(f, "UNSAFE"),
            SafetyVerdict::Unknown { reason, .. } => write!(f, "UNKNOWN ({})", reason),
        }
    }
}

// ---------------------------------------------------------------------------
// Safety Evidence
// ---------------------------------------------------------------------------

/// Evidence supporting a safety verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEvidence {
    pub kind: EvidenceKind,
    pub description: String,
    pub concentration_intervals: Vec<(DrugId, ConcentrationInterval)>,
    pub enzyme_intervals: Vec<(CypEnzyme, f64, f64)>,
    pub sub_evidence: Vec<SafetyEvidence>,
}

impl SafetyEvidence {
    pub fn new(kind: EvidenceKind, description: impl Into<String>) -> Self {
        Self {
            kind,
            description: description.into(),
            concentration_intervals: Vec::new(),
            enzyme_intervals: Vec::new(),
            sub_evidence: Vec::new(),
        }
    }

    pub fn empty() -> Self {
        Self::new(EvidenceKind::NoEvidence, "No evidence collected")
    }

    pub fn with_concentration(mut self, drug: DrugId, interval: ConcentrationInterval) -> Self {
        self.concentration_intervals.push((drug, interval));
        self
    }

    pub fn with_enzyme(mut self, enzyme: CypEnzyme, lo: f64, hi: f64) -> Self {
        self.enzyme_intervals.push((enzyme, lo, hi));
        self
    }

    pub fn combined(a: &SafetyEvidence, b: &SafetyEvidence) -> Self {
        let mut evidence = SafetyEvidence::new(
            EvidenceKind::Combined,
            "Combined evidence from multiple checks",
        );
        evidence.sub_evidence.push(a.clone());
        evidence.sub_evidence.push(b.clone());
        evidence
    }

    pub fn interval_within_bound(
        drug: DrugId,
        interval: ConcentrationInterval,
        bound_lo: f64,
        bound_hi: f64,
    ) -> Self {
        Self::new(
            EvidenceKind::IntervalContainment,
            format!(
                "Concentration interval [{:.4}, {:.4}] ⊆ [{:.4}, {:.4}]",
                interval.lo, interval.hi, bound_lo, bound_hi
            ),
        )
        .with_concentration(drug, interval)
    }

    pub fn interval_exceeds_bound(
        drug: DrugId,
        interval: ConcentrationInterval,
        bound_lo: f64,
        bound_hi: f64,
    ) -> Self {
        Self::new(
            EvidenceKind::IntervalViolation,
            format!(
                "Concentration interval [{:.4}, {:.4}] exceeds bound [{:.4}, {:.4}]",
                interval.lo, interval.hi, bound_lo, bound_hi
            ),
        )
        .with_concentration(drug, interval)
    }

    pub fn enzyme_within_bound(enzyme: CypEnzyme, activity_lo: f64, activity_hi: f64,
                                bound_lo: f64, bound_hi: f64) -> Self {
        Self::new(
            EvidenceKind::EnzymeContainment,
            format!(
                "Enzyme {} activity [{:.4}, {:.4}] ⊆ [{:.4}, {:.4}]",
                enzyme, activity_lo, activity_hi, bound_lo, bound_hi
            ),
        )
        .with_enzyme(enzyme, activity_lo, activity_hi)
    }

    pub fn enzyme_exceeds_bound(enzyme: CypEnzyme, activity_lo: f64, activity_hi: f64,
                                 bound_lo: f64, bound_hi: f64) -> Self {
        Self::new(
            EvidenceKind::EnzymeViolation,
            format!(
                "Enzyme {} activity [{:.4}, {:.4}] violates bound [{:.4}, {:.4}]",
                enzyme, activity_lo, activity_hi, bound_lo, bound_hi
            ),
        )
        .with_enzyme(enzyme, activity_lo, activity_hi)
    }
}

impl fmt::Display for SafetyEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}", self.kind, self.description)
    }
}

/// The kind of evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceKind {
    /// Interval is entirely within the safe bound.
    IntervalContainment,
    /// Interval entirely outside safe bound (proves violation).
    IntervalViolation,
    /// Interval partially overlaps bound (inconclusive).
    IntervalOverlap,
    /// Enzyme activity within bound.
    EnzymeContainment,
    /// Enzyme activity violates bound.
    EnzymeViolation,
    /// Interaction severity classification.
    InteractionSeverity,
    /// Combined evidence from multiple checks.
    Combined,
    /// No evidence collected.
    NoEvidence,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_property_concentration_bound() {
        let prop = SafetyProperty::concentration_bound(DrugId::new("warfarin"), 1.0, 4.0);
        assert!(prop.id.contains("warfarin"));
        let drugs = prop.referenced_drugs();
        assert!(drugs.contains(&DrugId::new("warfarin")));
    }

    #[test]
    fn test_safety_property_interaction_limit() {
        let prop = SafetyProperty::interaction_limit(
            DrugId::new("a"), DrugId::new("b"), Severity::Moderate,
        );
        let drugs = prop.referenced_drugs();
        assert_eq!(drugs.len(), 2);
    }

    #[test]
    fn test_safety_property_enzyme_bound() {
        let prop = SafetyProperty::enzyme_activity_bound(CypEnzyme::CYP3A4, 0.2, 1.5);
        let enzymes = prop.referenced_enzymes();
        assert!(enzymes.contains(&CypEnzyme::CYP3A4));
    }

    #[test]
    fn test_safety_property_conjunction() {
        let p1 = SafetyProperty::concentration_bound(DrugId::new("a"), 0.0, 10.0);
        let p2 = SafetyProperty::concentration_bound(DrugId::new("b"), 0.0, 5.0);
        let conj = SafetyProperty::conjunction(vec![p1, p2]);
        assert_eq!(conj.referenced_drugs().len(), 2);
    }

    #[test]
    fn test_safety_verdict_safe() {
        let v = SafetyVerdict::safe(SafetyEvidence::empty());
        assert!(v.is_safe());
        assert!(!v.is_unsafe());
        assert!(!v.needs_tier2());
    }

    #[test]
    fn test_safety_verdict_unsafe() {
        let v = SafetyVerdict::unsafe_verdict(SafetyEvidence::empty());
        assert!(v.is_unsafe());
        assert!(v.needs_tier2());
    }

    #[test]
    fn test_safety_verdict_unknown() {
        let v = SafetyVerdict::unknown("imprecise", SafetyEvidence::empty());
        assert!(v.is_unknown());
        assert!(v.needs_tier2());
    }

    #[test]
    fn test_verdict_combine_and() {
        let safe1 = SafetyVerdict::safe(SafetyEvidence::empty());
        let safe2 = SafetyVerdict::safe(SafetyEvidence::empty());
        let combined = SafetyVerdict::combine_and(&safe1, &safe2);
        assert!(combined.is_safe());

        let unsafe1 = SafetyVerdict::unsafe_verdict(SafetyEvidence::empty());
        let combined2 = SafetyVerdict::combine_and(&safe1, &unsafe1);
        assert!(combined2.is_unsafe());
    }

    #[test]
    fn test_verdict_combine_or() {
        let safe1 = SafetyVerdict::safe(SafetyEvidence::empty());
        let unsafe1 = SafetyVerdict::unsafe_verdict(SafetyEvidence::empty());
        let combined = SafetyVerdict::combine_or(&safe1, &unsafe1);
        assert!(combined.is_safe());

        let unsafe2 = SafetyVerdict::unsafe_verdict(SafetyEvidence::empty());
        let combined2 = SafetyVerdict::combine_or(&unsafe1, &unsafe2);
        assert!(combined2.is_unsafe());
    }

    #[test]
    fn test_evidence_construction() {
        let ev = SafetyEvidence::interval_within_bound(
            DrugId::new("test"),
            ConcentrationInterval::new(1.0, 3.0),
            0.0, 5.0,
        );
        assert_eq!(ev.kind, EvidenceKind::IntervalContainment);
        assert_eq!(ev.concentration_intervals.len(), 1);
    }

    #[test]
    fn test_evidence_combined() {
        let e1 = SafetyEvidence::empty();
        let e2 = SafetyEvidence::empty();
        let combined = SafetyEvidence::combined(&e1, &e2);
        assert_eq!(combined.kind, EvidenceKind::Combined);
        assert_eq!(combined.sub_evidence.len(), 2);
    }

    #[test]
    fn test_verdict_display() {
        let v = SafetyVerdict::safe(SafetyEvidence::empty());
        assert_eq!(format!("{}", v), "SAFE");
        let v = SafetyVerdict::unsafe_verdict(SafetyEvidence::empty());
        assert_eq!(format!("{}", v), "UNSAFE");
    }

    #[test]
    fn test_property_display() {
        let p = SafetyProperty::concentration_bound(DrugId::new("warfarin"), 1.0, 4.0);
        let s = format!("{}", p);
        assert!(s.contains("warfarin"));
    }
}

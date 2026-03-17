//! Guideline format definitions, structured document types and standard
//! templates that represent clinical practice guidelines in a machine-readable
//! form suitable for downstream formal-methods analysis.

use indexmap::IndexMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Core document types
// ---------------------------------------------------------------------------

/// A complete clinical practice guideline encoded as a structured document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineDocument {
    pub id: String,
    pub metadata: GuidelineMetadata,
    pub decision_points: Vec<DecisionPoint>,
    pub transitions: Vec<TransitionRule>,
    pub safety_constraints: Vec<SafetyConstraint>,
    pub monitoring: Vec<MonitoringRequirement>,
    /// Optional free-form annotations keyed by topic.
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

/// Bibliographic / administrative metadata for a guideline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineMetadata {
    pub title: String,
    pub version: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub publication_date: Option<String>,
    #[serde(default)]
    pub source_organization: Option<String>,
    #[serde(default)]
    pub condition: Option<String>,
    #[serde(default)]
    pub evidence_level: Option<EvidenceLevel>,
    #[serde(default)]
    pub supersedes: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Levels of clinical evidence (simplified GRADE-like).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceLevel {
    High,
    Moderate,
    Low,
    VeryLow,
    ExpertOpinion,
}

impl std::fmt::Display for EvidenceLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::High => write!(f, "High"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Low => write!(f, "Low"),
            Self::VeryLow => write!(f, "VeryLow"),
            Self::ExpertOpinion => write!(f, "ExpertOpinion"),
        }
    }
}

// ---------------------------------------------------------------------------
// Decision points & branches
// ---------------------------------------------------------------------------

/// A node in the guideline decision tree / graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionPoint {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
    pub branches: Vec<Branch>,
    #[serde(default)]
    pub is_initial: bool,
    #[serde(default)]
    pub is_terminal: bool,
    #[serde(default)]
    pub invariants: Vec<String>,
    #[serde(default)]
    pub urgency: Option<Urgency>,
}

/// Urgency of a decision point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Urgency {
    Routine,
    Urgent,
    Emergent,
}

/// A branch from a decision point representing one possible clinical pathway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub id: String,
    pub guard: GuidelineGuard,
    pub actions: Vec<GuidelineAction>,
    pub target: String,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub evidence_level: Option<EvidenceLevel>,
    #[serde(default)]
    pub notes: Option<String>,
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/// Clinical actions that may be taken at a decision point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GuidelineAction {
    StartMedication {
        medication: String,
        dose: DoseSpec,
        route: String,
        #[serde(default)]
        reason: Option<String>,
    },
    AdjustDose {
        medication: String,
        new_dose: DoseSpec,
        #[serde(default)]
        reason: Option<String>,
    },
    StopMedication {
        medication: String,
        #[serde(default)]
        taper: Option<TaperSchedule>,
        #[serde(default)]
        reason: Option<String>,
    },
    SwitchMedication {
        from_medication: String,
        to_medication: String,
        new_dose: DoseSpec,
        #[serde(default)]
        reason: Option<String>,
    },
    OrderLab {
        test_name: String,
        #[serde(default)]
        urgency: Option<Urgency>,
        #[serde(default)]
        repeat_interval_days: Option<u32>,
    },
    MonitorInterval {
        parameter: String,
        interval_days: u32,
        #[serde(default)]
        duration_days: Option<u32>,
    },
    Refer {
        specialty: String,
        #[serde(default)]
        urgency: Option<Urgency>,
        #[serde(default)]
        reason: Option<String>,
    },
    LifestyleModification {
        category: String,
        description: String,
    },
    PatientEducation {
        topic: String,
        #[serde(default)]
        materials: Vec<String>,
    },
    SetClock {
        clock_name: String,
        value: f64,
    },
    ResetClock {
        clock_name: String,
    },
    RecordOutcome {
        outcome: String,
        #[serde(default)]
        value: Option<String>,
    },
    CombinationTherapy {
        medications: Vec<MedicationSpec>,
        #[serde(default)]
        reason: Option<String>,
    },
    EmergencyIntervention {
        description: String,
        #[serde(default)]
        medications: Vec<MedicationSpec>,
    },
    Reassess {
        interval_days: u32,
        #[serde(default)]
        criteria: Vec<String>,
    },
    NoChange {
        #[serde(default)]
        reason: Option<String>,
    },
}

impl GuidelineAction {
    /// Return every medication name referenced by this action.
    pub fn referenced_medications(&self) -> Vec<&str> {
        match self {
            Self::StartMedication { medication, .. } => vec![medication.as_str()],
            Self::AdjustDose { medication, .. } => vec![medication.as_str()],
            Self::StopMedication { medication, .. } => vec![medication.as_str()],
            Self::SwitchMedication {
                from_medication,
                to_medication,
                ..
            } => vec![from_medication.as_str(), to_medication.as_str()],
            Self::CombinationTherapy { medications, .. }
            | Self::EmergencyIntervention { medications, .. } => {
                medications.iter().map(|m| m.name.as_str()).collect()
            }
            _ => vec![],
        }
    }

    /// True when the action prescribes or modifies a medication.
    pub fn is_medication_action(&self) -> bool {
        matches!(
            self,
            Self::StartMedication { .. }
                | Self::AdjustDose { .. }
                | Self::StopMedication { .. }
                | Self::SwitchMedication { .. }
                | Self::CombinationTherapy { .. }
                | Self::EmergencyIntervention { .. }
        )
    }

    /// True when the action involves ordering a lab or setting up monitoring.
    pub fn is_monitoring_action(&self) -> bool {
        matches!(
            self,
            Self::OrderLab { .. } | Self::MonitorInterval { .. } | Self::Reassess { .. }
        )
    }

    /// A short human-readable description.
    pub fn summary(&self) -> String {
        match self {
            Self::StartMedication { medication, dose, .. } => {
                format!("Start {} {}", medication, dose)
            }
            Self::AdjustDose { medication, new_dose, .. } => {
                format!("Adjust {} to {}", medication, new_dose)
            }
            Self::StopMedication { medication, .. } => format!("Stop {}", medication),
            Self::SwitchMedication {
                from_medication,
                to_medication,
                ..
            } => format!("Switch {} → {}", from_medication, to_medication),
            Self::OrderLab { test_name, .. } => format!("Order lab: {}", test_name),
            Self::MonitorInterval { parameter, interval_days, .. } => {
                format!("Monitor {} q{}d", parameter, interval_days)
            }
            Self::Refer { specialty, .. } => format!("Refer to {}", specialty),
            Self::LifestyleModification { category, .. } => {
                format!("Lifestyle: {}", category)
            }
            Self::PatientEducation { topic, .. } => format!("Educate: {}", topic),
            Self::SetClock { clock_name, value } => {
                format!("Set clock {} = {}", clock_name, value)
            }
            Self::ResetClock { clock_name } => format!("Reset clock {}", clock_name),
            Self::RecordOutcome { outcome, .. } => format!("Record: {}", outcome),
            Self::CombinationTherapy { medications, .. } => {
                let names: Vec<&str> = medications.iter().map(|m| m.name.as_str()).collect();
                format!("Combination: {}", names.join(" + "))
            }
            Self::EmergencyIntervention { description, .. } => {
                format!("Emergency: {}", description)
            }
            Self::Reassess { interval_days, .. } => {
                format!("Reassess in {} days", interval_days)
            }
            Self::NoChange { .. } => "No change".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Dose / medication helpers
// ---------------------------------------------------------------------------

/// A single medication inside combination therapy or emergency actions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MedicationSpec {
    pub name: String,
    pub dose: DoseSpec,
    #[serde(default)]
    pub route: Option<String>,
}

/// Dose specification with value, unit, and optional range.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DoseSpec {
    pub value: f64,
    pub unit: String,
    #[serde(default)]
    pub frequency: Option<String>,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
}

impl std::fmt::Display for DoseSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.value, self.unit)?;
        if let Some(freq) = &self.frequency {
            write!(f, " {}", freq)?;
        }
        Ok(())
    }
}

impl DoseSpec {
    pub fn new(value: f64, unit: &str) -> Self {
        Self {
            value,
            unit: unit.to_string(),
            frequency: None,
            min: None,
            max: None,
        }
    }

    pub fn with_frequency(mut self, freq: &str) -> Self {
        self.frequency = Some(freq.to_string());
        self
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// True when the dose falls within the declared min/max range (if set).
    pub fn is_in_range(&self) -> bool {
        if let (Some(lo), Some(hi)) = (self.min, self.max) {
            self.value >= lo && self.value <= hi
        } else {
            true
        }
    }
}

/// Taper schedule for stopping a medication gradually.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaperSchedule {
    pub steps: Vec<TaperStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaperStep {
    pub dose: DoseSpec,
    pub duration_days: u32,
}

impl TaperSchedule {
    /// Total duration of the taper in days.
    pub fn total_days(&self) -> u32 {
        self.steps.iter().map(|s| s.duration_days).sum()
    }

    /// True when the taper is monotonically non-increasing.
    pub fn is_monotonic_decreasing(&self) -> bool {
        self.steps
            .windows(2)
            .all(|w| w[0].dose.value >= w[1].dose.value)
    }
}

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

/// Condition that must be satisfied to traverse a branch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GuidelineGuard {
    ClinicalPredicate {
        parameter: String,
        operator: ComparisonOp,
        threshold: f64,
        #[serde(default)]
        unit: Option<String>,
    },
    LabThreshold {
        test_name: String,
        operator: ComparisonOp,
        value: f64,
        #[serde(default)]
        unit: Option<String>,
    },
    TimeElapsed {
        clock: String,
        operator: ComparisonOp,
        days: f64,
    },
    ConcentrationRange {
        drug: String,
        min: f64,
        max: f64,
        #[serde(default)]
        unit: Option<String>,
    },
    DiagnosisPresent {
        diagnosis: String,
    },
    MedicationActive {
        medication: String,
    },
    AgeRange {
        min_years: Option<f64>,
        max_years: Option<f64>,
    },
    AllergyPresent {
        substance: String,
    },
    And(Vec<GuidelineGuard>),
    Or(Vec<GuidelineGuard>),
    Not(Box<GuidelineGuard>),
    True,
    False,
}

impl GuidelineGuard {
    /// Logical AND of two guards.
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, g) | (g, Self::True) => g,
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::And(mut a), Self::And(b)) => {
                a.extend(b);
                Self::And(a)
            }
            (Self::And(mut a), g) => {
                a.push(g);
                Self::And(a)
            }
            (g, Self::And(mut a)) => {
                a.insert(0, g);
                Self::And(a)
            }
            (a, b) => Self::And(vec![a, b]),
        }
    }

    /// Logical OR of two guards.
    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::True, _) | (_, Self::True) => Self::True,
            (Self::False, g) | (g, Self::False) => g,
            (Self::Or(mut a), Self::Or(b)) => {
                a.extend(b);
                Self::Or(a)
            }
            (Self::Or(mut a), g) => {
                a.push(g);
                Self::Or(a)
            }
            (g, Self::Or(mut a)) => {
                a.insert(0, g);
                Self::Or(a)
            }
            (a, b) => Self::Or(vec![a, b]),
        }
    }

    /// Logical NOT.
    pub fn negate(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Not(inner) => *inner,
            other => Self::Not(Box::new(other)),
        }
    }

    /// Collect all parameter / test names referenced by this guard.
    pub fn referenced_parameters(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.collect_params(&mut out);
        out
    }

    fn collect_params(&self, acc: &mut Vec<String>) {
        match self {
            Self::ClinicalPredicate { parameter, .. } => acc.push(parameter.clone()),
            Self::LabThreshold { test_name, .. } => acc.push(test_name.clone()),
            Self::TimeElapsed { clock, .. } => acc.push(clock.clone()),
            Self::ConcentrationRange { drug, .. } => acc.push(drug.clone()),
            Self::DiagnosisPresent { diagnosis } => acc.push(diagnosis.clone()),
            Self::MedicationActive { medication } => acc.push(medication.clone()),
            Self::AgeRange { .. } => acc.push("age".to_string()),
            Self::AllergyPresent { substance } => acc.push(substance.clone()),
            Self::And(gs) | Self::Or(gs) => {
                for g in gs {
                    g.collect_params(acc);
                }
            }
            Self::Not(g) => g.collect_params(acc),
            Self::True | Self::False => {}
        }
    }

    /// Depth of the expression tree.
    pub fn depth(&self) -> usize {
        match self {
            Self::And(gs) | Self::Or(gs) => 1 + gs.iter().map(|g| g.depth()).max().unwrap_or(0),
            Self::Not(g) => 1 + g.depth(),
            _ => 1,
        }
    }

    /// Number of leaf predicates.
    pub fn leaf_count(&self) -> usize {
        match self {
            Self::And(gs) | Self::Or(gs) => gs.iter().map(|g| g.leaf_count()).sum(),
            Self::Not(g) => g.leaf_count(),
            Self::True | Self::False => 0,
            _ => 1,
        }
    }

    /// Returns true if the guard is a simple leaf (not a combinator).
    pub fn is_leaf(&self) -> bool {
        !matches!(self, Self::And(_) | Self::Or(_) | Self::Not(_))
    }

    /// Replace all occurrences of a parameter name.
    pub fn rename_parameter(&mut self, from: &str, to: &str) {
        match self {
            Self::ClinicalPredicate { parameter, .. } if parameter == from => {
                *parameter = to.to_string();
            }
            Self::LabThreshold { test_name, .. } if test_name == from => {
                *test_name = to.to_string();
            }
            Self::TimeElapsed { clock, .. } if clock == from => {
                *clock = to.to_string();
            }
            Self::ConcentrationRange { drug, .. } if drug == from => {
                *drug = to.to_string();
            }
            Self::DiagnosisPresent { diagnosis } if diagnosis == from => {
                *diagnosis = to.to_string();
            }
            Self::MedicationActive { medication } if medication == from => {
                *medication = to.to_string();
            }
            Self::AllergyPresent { substance } if substance == from => {
                *substance = to.to_string();
            }
            Self::And(gs) | Self::Or(gs) => {
                for g in gs {
                    g.rename_parameter(from, to);
                }
            }
            Self::Not(g) => g.rename_parameter(from, to),
            _ => {}
        }
    }
}

/// Comparison operators used in guard predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
    Ne,
}

impl ComparisonOp {
    /// Evaluate `lhs <op> rhs`.
    pub fn evaluate(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            Self::Lt => lhs < rhs,
            Self::Le => lhs <= rhs,
            Self::Eq => (lhs - rhs).abs() < f64::EPSILON,
            Self::Ge => lhs >= rhs,
            Self::Gt => lhs > rhs,
            Self::Ne => (lhs - rhs).abs() >= f64::EPSILON,
        }
    }

    /// Flip the operator direction (e.g. `Lt` → `Gt`).
    pub fn flip(&self) -> Self {
        match self {
            Self::Lt => Self::Gt,
            Self::Le => Self::Ge,
            Self::Eq => Self::Eq,
            Self::Ge => Self::Le,
            Self::Gt => Self::Lt,
            Self::Ne => Self::Ne,
        }
    }

    /// Negate the operator (e.g. `Lt` → `Ge`).
    pub fn negate(&self) -> Self {
        match self {
            Self::Lt => Self::Ge,
            Self::Le => Self::Gt,
            Self::Eq => Self::Ne,
            Self::Ge => Self::Lt,
            Self::Gt => Self::Le,
            Self::Ne => Self::Eq,
        }
    }
}

impl std::fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lt => write!(f, "<"),
            Self::Le => write!(f, "<="),
            Self::Eq => write!(f, "=="),
            Self::Ge => write!(f, ">="),
            Self::Gt => write!(f, ">"),
            Self::Ne => write!(f, "!="),
        }
    }
}

// ---------------------------------------------------------------------------
// Transitions
// ---------------------------------------------------------------------------

/// Explicit transition rule between two decision points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRule {
    pub id: String,
    pub source: String,
    pub target: String,
    pub guard: GuidelineGuard,
    pub actions: Vec<GuidelineAction>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub weight: Option<f64>,
    #[serde(default)]
    pub description: Option<String>,
}

// ---------------------------------------------------------------------------
// Safety constraints & monitoring
// ---------------------------------------------------------------------------

/// A safety constraint that must hold across the entire guideline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraint {
    pub id: String,
    pub description: String,
    pub guard: GuidelineGuard,
    #[serde(default)]
    pub severity: ConstraintSeverity,
    #[serde(default)]
    pub applies_to: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintSeverity {
    Critical,
    Warning,
    Info,
}

impl Default for ConstraintSeverity {
    fn default() -> Self {
        Self::Warning
    }
}

/// Monitoring requirement (lab, vital, etc.) with scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringRequirement {
    pub id: String,
    pub parameter: String,
    pub interval_days: u32,
    #[serde(default)]
    pub duration_days: Option<u32>,
    #[serde(default)]
    pub target_range: Option<(f64, f64)>,
    #[serde(default)]
    pub alert_threshold: Option<f64>,
    #[serde(default)]
    pub applies_to_states: Vec<String>,
}

// ---------------------------------------------------------------------------
// GuidelineDocument helpers
// ---------------------------------------------------------------------------

impl GuidelineDocument {
    /// Create a new empty guideline document with the given title.
    pub fn new(title: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            metadata: GuidelineMetadata {
                title: title.to_string(),
                version: "1.0.0".to_string(),
                authors: Vec::new(),
                publication_date: None,
                source_organization: None,
                condition: None,
                evidence_level: None,
                supersedes: None,
                tags: Vec::new(),
            },
            decision_points: Vec::new(),
            transitions: Vec::new(),
            safety_constraints: Vec::new(),
            monitoring: Vec::new(),
            annotations: HashMap::new(),
        }
    }

    /// Number of decision points.
    pub fn num_decision_points(&self) -> usize {
        self.decision_points.len()
    }

    /// Return the initial decision points.
    pub fn initial_points(&self) -> Vec<&DecisionPoint> {
        self.decision_points
            .iter()
            .filter(|dp| dp.is_initial)
            .collect()
    }

    /// Return the terminal decision points.
    pub fn terminal_points(&self) -> Vec<&DecisionPoint> {
        self.decision_points
            .iter()
            .filter(|dp| dp.is_terminal)
            .collect()
    }

    /// Collect all unique medication names referenced anywhere.
    pub fn all_medications(&self) -> Vec<String> {
        let mut meds = std::collections::HashSet::new();
        for dp in &self.decision_points {
            for br in &dp.branches {
                for act in &br.actions {
                    for m in act.referenced_medications() {
                        meds.insert(m.to_string());
                    }
                }
            }
        }
        for tr in &self.transitions {
            for act in &tr.actions {
                for m in act.referenced_medications() {
                    meds.insert(m.to_string());
                }
            }
        }
        let mut v: Vec<String> = meds.into_iter().collect();
        v.sort();
        v
    }

    /// Collect all decision point IDs.
    pub fn decision_point_ids(&self) -> Vec<&str> {
        self.decision_points.iter().map(|dp| dp.id.as_str()).collect()
    }

    /// Find a decision point by ID.
    pub fn find_decision_point(&self, id: &str) -> Option<&DecisionPoint> {
        self.decision_points.iter().find(|dp| dp.id == id)
    }

    /// Find a transition by ID.
    pub fn find_transition(&self, id: &str) -> Option<&TransitionRule> {
        self.transitions.iter().find(|tr| tr.id == id)
    }

    /// Total number of branches across all decision points.
    pub fn total_branches(&self) -> usize {
        self.decision_points.iter().map(|dp| dp.branches.len()).sum()
    }

    /// All unique parameters referenced by all guards.
    pub fn all_parameters(&self) -> Vec<String> {
        let mut params = std::collections::HashSet::new();
        for dp in &self.decision_points {
            for br in &dp.branches {
                for p in br.guard.referenced_parameters() {
                    params.insert(p);
                }
            }
        }
        for tr in &self.transitions {
            for p in tr.guard.referenced_parameters() {
                params.insert(p);
            }
        }
        let mut v: Vec<String> = params.into_iter().collect();
        v.sort();
        v
    }

    /// Build an adjacency list representation (decision-point-id → [target-ids]).
    pub fn adjacency(&self) -> HashMap<String, Vec<String>> {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for dp in &self.decision_points {
            let entry = adj.entry(dp.id.clone()).or_default();
            for br in &dp.branches {
                if !entry.contains(&br.target) {
                    entry.push(br.target.clone());
                }
            }
        }
        for tr in &self.transitions {
            let entry = adj.entry(tr.source.clone()).or_default();
            if !entry.contains(&tr.target) {
                entry.push(tr.target.clone());
            }
        }
        adj
    }

    /// Add a decision point to the document.
    pub fn add_decision_point(&mut self, dp: DecisionPoint) {
        self.decision_points.push(dp);
    }

    /// Add a transition rule.
    pub fn add_transition(&mut self, tr: TransitionRule) {
        self.transitions.push(tr);
    }

    /// Add a safety constraint.
    pub fn add_safety_constraint(&mut self, sc: SafetyConstraint) {
        self.safety_constraints.push(sc);
    }

    /// Add a monitoring requirement.
    pub fn add_monitoring(&mut self, mr: MonitoringRequirement) {
        self.monitoring.push(mr);
    }
}

// ---------------------------------------------------------------------------
// Standard template: minimal diabetes guideline (for quick testing)
// ---------------------------------------------------------------------------

/// Build a small but complete type-2 diabetes guideline for testing purposes.
pub fn standard_diabetes_template() -> GuidelineDocument {
    let mut doc = GuidelineDocument::new("Type 2 Diabetes Management");
    doc.metadata.condition = Some("Type 2 Diabetes".to_string());
    doc.metadata.version = "2.0.0".to_string();
    doc.metadata.source_organization = Some("ADA".to_string());
    doc.metadata.evidence_level = Some(EvidenceLevel::High);
    doc.metadata.tags = vec!["diabetes".into(), "endocrine".into()];

    // Initial assessment
    doc.add_decision_point(DecisionPoint {
        id: "initial_assessment".into(),
        label: "Initial Assessment".into(),
        description: Some("Assess HbA1c and comorbidities".into()),
        branches: vec![
            Branch {
                id: "mild".into(),
                guard: GuidelineGuard::LabThreshold {
                    test_name: "HbA1c".into(),
                    operator: ComparisonOp::Lt,
                    value: 7.5,
                    unit: Some("%".into()),
                },
                actions: vec![GuidelineAction::StartMedication {
                    medication: "metformin".into(),
                    dose: DoseSpec::new(500.0, "mg").with_frequency("BID"),
                    route: "oral".into(),
                    reason: Some("First-line therapy".into()),
                }],
                target: "metformin_monotherapy".into(),
                priority: 1,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "moderate".into(),
                guard: GuidelineGuard::And(vec![
                    GuidelineGuard::LabThreshold {
                        test_name: "HbA1c".into(),
                        operator: ComparisonOp::Ge,
                        value: 7.5,
                        unit: Some("%".into()),
                    },
                    GuidelineGuard::LabThreshold {
                        test_name: "HbA1c".into(),
                        operator: ComparisonOp::Lt,
                        value: 9.0,
                        unit: Some("%".into()),
                    },
                ]),
                actions: vec![GuidelineAction::StartMedication {
                    medication: "metformin".into(),
                    dose: DoseSpec::new(1000.0, "mg").with_frequency("BID"),
                    route: "oral".into(),
                    reason: Some("Higher dose for moderate HbA1c".into()),
                }],
                target: "dual_therapy_evaluation".into(),
                priority: 2,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "severe".into(),
                guard: GuidelineGuard::LabThreshold {
                    test_name: "HbA1c".into(),
                    operator: ComparisonOp::Ge,
                    value: 9.0,
                    unit: Some("%".into()),
                },
                actions: vec![GuidelineAction::CombinationTherapy {
                    medications: vec![
                        MedicationSpec {
                            name: "metformin".into(),
                            dose: DoseSpec::new(1000.0, "mg").with_frequency("BID"),
                            route: Some("oral".into()),
                        },
                        MedicationSpec {
                            name: "insulin_glargine".into(),
                            dose: DoseSpec::new(10.0, "units").with_frequency("QHS"),
                            route: Some("subcutaneous".into()),
                        },
                    ],
                    reason: Some("Severe hyperglycemia requires combination".into()),
                }],
                target: "insulin_titration".into(),
                priority: 3,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
        ],
        is_initial: true,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    // Metformin monotherapy
    doc.add_decision_point(DecisionPoint {
        id: "metformin_monotherapy".into(),
        label: "Metformin Monotherapy".into(),
        description: Some("Monitor response to metformin".into()),
        branches: vec![
            Branch {
                id: "on_target".into(),
                guard: GuidelineGuard::LabThreshold {
                    test_name: "HbA1c".into(),
                    operator: ComparisonOp::Lt,
                    value: 7.0,
                    unit: Some("%".into()),
                },
                actions: vec![GuidelineAction::Reassess {
                    interval_days: 90,
                    criteria: vec!["HbA1c".into(), "renal_function".into()],
                }],
                target: "maintenance".into(),
                priority: 1,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "above_target".into(),
                guard: GuidelineGuard::LabThreshold {
                    test_name: "HbA1c".into(),
                    operator: ComparisonOp::Ge,
                    value: 7.0,
                    unit: Some("%".into()),
                },
                actions: vec![],
                target: "dual_therapy_evaluation".into(),
                priority: 2,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
        ],
        is_initial: false,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    // Dual-therapy evaluation
    doc.add_decision_point(DecisionPoint {
        id: "dual_therapy_evaluation".into(),
        label: "Dual Therapy Evaluation".into(),
        description: Some("Choose second agent".into()),
        branches: vec![
            Branch {
                id: "add_sglt2".into(),
                guard: GuidelineGuard::And(vec![
                    GuidelineGuard::ClinicalPredicate {
                        parameter: "eGFR".into(),
                        operator: ComparisonOp::Ge,
                        threshold: 30.0,
                        unit: Some("mL/min/1.73m2".into()),
                    },
                    GuidelineGuard::DiagnosisPresent {
                        diagnosis: "heart_failure".into(),
                    },
                ]),
                actions: vec![GuidelineAction::StartMedication {
                    medication: "empagliflozin".into(),
                    dose: DoseSpec::new(10.0, "mg").with_frequency("QD"),
                    route: "oral".into(),
                    reason: Some("CV benefit in HF".into()),
                }],
                target: "dual_therapy_monitoring".into(),
                priority: 1,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "add_glp1".into(),
                guard: GuidelineGuard::DiagnosisPresent {
                    diagnosis: "atherosclerotic_cvd".into(),
                },
                actions: vec![GuidelineAction::StartMedication {
                    medication: "semaglutide".into(),
                    dose: DoseSpec::new(0.25, "mg").with_frequency("weekly"),
                    route: "subcutaneous".into(),
                    reason: Some("CV benefit in ASCVD".into()),
                }],
                target: "dual_therapy_monitoring".into(),
                priority: 2,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "add_dpp4".into(),
                guard: GuidelineGuard::True,
                actions: vec![GuidelineAction::StartMedication {
                    medication: "sitagliptin".into(),
                    dose: DoseSpec::new(100.0, "mg").with_frequency("QD"),
                    route: "oral".into(),
                    reason: Some("Well tolerated second-line".into()),
                }],
                target: "dual_therapy_monitoring".into(),
                priority: 3,
                evidence_level: Some(EvidenceLevel::Moderate),
                notes: None,
            },
        ],
        is_initial: false,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    // Dual-therapy monitoring
    doc.add_decision_point(DecisionPoint {
        id: "dual_therapy_monitoring".into(),
        label: "Dual Therapy Monitoring".into(),
        description: Some("Monitor dual therapy response".into()),
        branches: vec![
            Branch {
                id: "controlled".into(),
                guard: GuidelineGuard::LabThreshold {
                    test_name: "HbA1c".into(),
                    operator: ComparisonOp::Lt,
                    value: 7.0,
                    unit: Some("%".into()),
                },
                actions: vec![GuidelineAction::Reassess {
                    interval_days: 90,
                    criteria: vec!["HbA1c".into()],
                }],
                target: "maintenance".into(),
                priority: 1,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "uncontrolled".into(),
                guard: GuidelineGuard::LabThreshold {
                    test_name: "HbA1c".into(),
                    operator: ComparisonOp::Ge,
                    value: 7.0,
                    unit: Some("%".into()),
                },
                actions: vec![],
                target: "insulin_titration".into(),
                priority: 2,
                evidence_level: Some(EvidenceLevel::Moderate),
                notes: None,
            },
        ],
        is_initial: false,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    // Insulin titration
    doc.add_decision_point(DecisionPoint {
        id: "insulin_titration".into(),
        label: "Insulin Titration".into(),
        description: Some("Titrate basal insulin".into()),
        branches: vec![
            Branch {
                id: "fasting_high".into(),
                guard: GuidelineGuard::LabThreshold {
                    test_name: "fasting_glucose".into(),
                    operator: ComparisonOp::Gt,
                    value: 130.0,
                    unit: Some("mg/dL".into()),
                },
                actions: vec![GuidelineAction::AdjustDose {
                    medication: "insulin_glargine".into(),
                    new_dose: DoseSpec::new(2.0, "units").with_frequency("increase"),
                    reason: Some("Fasting BG above target".into()),
                }],
                target: "insulin_titration".into(),
                priority: 1,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "fasting_low".into(),
                guard: GuidelineGuard::LabThreshold {
                    test_name: "fasting_glucose".into(),
                    operator: ComparisonOp::Lt,
                    value: 70.0,
                    unit: Some("mg/dL".into()),
                },
                actions: vec![GuidelineAction::AdjustDose {
                    medication: "insulin_glargine".into(),
                    new_dose: DoseSpec::new(-4.0, "units").with_frequency("decrease"),
                    reason: Some("Hypoglycemia risk".into()),
                }],
                target: "insulin_titration".into(),
                priority: 1,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "fasting_target".into(),
                guard: GuidelineGuard::And(vec![
                    GuidelineGuard::LabThreshold {
                        test_name: "fasting_glucose".into(),
                        operator: ComparisonOp::Ge,
                        value: 70.0,
                        unit: Some("mg/dL".into()),
                    },
                    GuidelineGuard::LabThreshold {
                        test_name: "fasting_glucose".into(),
                        operator: ComparisonOp::Le,
                        value: 130.0,
                        unit: Some("mg/dL".into()),
                    },
                ]),
                actions: vec![GuidelineAction::Reassess {
                    interval_days: 90,
                    criteria: vec!["HbA1c".into(), "fasting_glucose".into()],
                }],
                target: "maintenance".into(),
                priority: 2,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
        ],
        is_initial: false,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    // Maintenance
    doc.add_decision_point(DecisionPoint {
        id: "maintenance".into(),
        label: "Maintenance".into(),
        description: Some("Stable glycemic control".into()),
        branches: vec![Branch {
            id: "continue".into(),
            guard: GuidelineGuard::True,
            actions: vec![GuidelineAction::Reassess {
                interval_days: 180,
                criteria: vec!["HbA1c".into(), "renal_function".into(), "lipids".into()],
            }],
            target: "maintenance".into(),
            priority: 1,
            evidence_level: Some(EvidenceLevel::High),
            notes: None,
        }],
        is_initial: false,
        is_terminal: true,
        invariants: vec![],
        urgency: None,
    });

    // Safety constraints
    doc.add_safety_constraint(SafetyConstraint {
        id: "no_metformin_low_egfr".into(),
        description: "Metformin contraindicated when eGFR < 30".into(),
        guard: GuidelineGuard::And(vec![
            GuidelineGuard::MedicationActive {
                medication: "metformin".into(),
            },
            GuidelineGuard::ClinicalPredicate {
                parameter: "eGFR".into(),
                operator: ComparisonOp::Lt,
                threshold: 30.0,
                unit: Some("mL/min/1.73m2".into()),
            },
        ]),
        severity: ConstraintSeverity::Critical,
        applies_to: vec![],
    });

    doc.add_safety_constraint(SafetyConstraint {
        id: "hypoglycemia_monitoring".into(),
        description: "Monitor for hypoglycemia with insulin".into(),
        guard: GuidelineGuard::MedicationActive {
            medication: "insulin_glargine".into(),
        },
        severity: ConstraintSeverity::Warning,
        applies_to: vec!["insulin_titration".into()],
    });

    // Monitoring
    doc.add_monitoring(MonitoringRequirement {
        id: "hba1c_monitoring".into(),
        parameter: "HbA1c".into(),
        interval_days: 90,
        duration_days: None,
        target_range: Some((6.5, 7.0)),
        alert_threshold: Some(9.0),
        applies_to_states: vec![],
    });

    doc.add_monitoring(MonitoringRequirement {
        id: "renal_monitoring".into(),
        parameter: "eGFR".into(),
        interval_days: 180,
        duration_days: None,
        target_range: Some((60.0, 120.0)),
        alert_threshold: Some(30.0),
        applies_to_states: vec![],
    });

    doc
}

/// Build a small hypertension guideline for testing.
pub fn standard_hypertension_template() -> GuidelineDocument {
    let mut doc = GuidelineDocument::new("Hypertension Management (JNC-8 simplified)");
    doc.metadata.condition = Some("Hypertension".to_string());
    doc.metadata.version = "1.0.0".to_string();
    doc.metadata.source_organization = Some("JNC".to_string());

    doc.add_decision_point(DecisionPoint {
        id: "bp_assessment".into(),
        label: "BP Assessment".into(),
        description: Some("Classify blood pressure".into()),
        branches: vec![
            Branch {
                id: "normal_bp".into(),
                guard: GuidelineGuard::ClinicalPredicate {
                    parameter: "systolic_bp".into(),
                    operator: ComparisonOp::Lt,
                    threshold: 130.0,
                    unit: Some("mmHg".into()),
                },
                actions: vec![GuidelineAction::LifestyleModification {
                    category: "diet".into(),
                    description: "DASH diet".into(),
                }],
                target: "lifestyle".into(),
                priority: 1,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "stage1".into(),
                guard: GuidelineGuard::And(vec![
                    GuidelineGuard::ClinicalPredicate {
                        parameter: "systolic_bp".into(),
                        operator: ComparisonOp::Ge,
                        threshold: 130.0,
                        unit: Some("mmHg".into()),
                    },
                    GuidelineGuard::ClinicalPredicate {
                        parameter: "systolic_bp".into(),
                        operator: ComparisonOp::Lt,
                        threshold: 140.0,
                        unit: Some("mmHg".into()),
                    },
                ]),
                actions: vec![GuidelineAction::StartMedication {
                    medication: "lisinopril".into(),
                    dose: DoseSpec::new(10.0, "mg").with_frequency("QD"),
                    route: "oral".into(),
                    reason: Some("Stage 1 HTN".into()),
                }],
                target: "monotherapy_monitor".into(),
                priority: 2,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
            Branch {
                id: "stage2".into(),
                guard: GuidelineGuard::ClinicalPredicate {
                    parameter: "systolic_bp".into(),
                    operator: ComparisonOp::Ge,
                    threshold: 140.0,
                    unit: Some("mmHg".into()),
                },
                actions: vec![GuidelineAction::CombinationTherapy {
                    medications: vec![
                        MedicationSpec {
                            name: "lisinopril".into(),
                            dose: DoseSpec::new(20.0, "mg").with_frequency("QD"),
                            route: Some("oral".into()),
                        },
                        MedicationSpec {
                            name: "amlodipine".into(),
                            dose: DoseSpec::new(5.0, "mg").with_frequency("QD"),
                            route: Some("oral".into()),
                        },
                    ],
                    reason: Some("Stage 2 HTN requires dual therapy".into()),
                }],
                target: "dual_monitor".into(),
                priority: 3,
                evidence_level: Some(EvidenceLevel::High),
                notes: None,
            },
        ],
        is_initial: true,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    doc.add_decision_point(DecisionPoint {
        id: "lifestyle".into(),
        label: "Lifestyle Only".into(),
        description: None,
        branches: vec![Branch {
            id: "reassess_lifestyle".into(),
            guard: GuidelineGuard::TimeElapsed {
                clock: "treatment_clock".into(),
                operator: ComparisonOp::Ge,
                days: 90.0,
            },
            actions: vec![GuidelineAction::Reassess {
                interval_days: 90,
                criteria: vec!["blood_pressure".into()],
            }],
            target: "bp_assessment".into(),
            priority: 1,
            evidence_level: None,
            notes: None,
        }],
        is_initial: false,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    doc.add_decision_point(DecisionPoint {
        id: "monotherapy_monitor".into(),
        label: "Monotherapy Monitoring".into(),
        description: None,
        branches: vec![
            Branch {
                id: "controlled_mono".into(),
                guard: GuidelineGuard::ClinicalPredicate {
                    parameter: "systolic_bp".into(),
                    operator: ComparisonOp::Lt,
                    threshold: 130.0,
                    unit: Some("mmHg".into()),
                },
                actions: vec![GuidelineAction::NoChange { reason: Some("BP at target".into()) }],
                target: "stable".into(),
                priority: 1,
                evidence_level: None,
                notes: None,
            },
            Branch {
                id: "uncontrolled_mono".into(),
                guard: GuidelineGuard::ClinicalPredicate {
                    parameter: "systolic_bp".into(),
                    operator: ComparisonOp::Ge,
                    threshold: 130.0,
                    unit: Some("mmHg".into()),
                },
                actions: vec![GuidelineAction::AdjustDose {
                    medication: "lisinopril".into(),
                    new_dose: DoseSpec::new(20.0, "mg").with_frequency("QD"),
                    reason: Some("Uptitrate for BP control".into()),
                }],
                target: "dual_monitor".into(),
                priority: 2,
                evidence_level: None,
                notes: None,
            },
        ],
        is_initial: false,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    doc.add_decision_point(DecisionPoint {
        id: "dual_monitor".into(),
        label: "Dual Therapy Monitoring".into(),
        description: None,
        branches: vec![Branch {
            id: "dual_check".into(),
            guard: GuidelineGuard::True,
            actions: vec![GuidelineAction::MonitorInterval {
                parameter: "blood_pressure".into(),
                interval_days: 30,
                duration_days: Some(180),
            }],
            target: "stable".into(),
            priority: 1,
            evidence_level: None,
            notes: None,
        }],
        is_initial: false,
        is_terminal: false,
        invariants: vec![],
        urgency: None,
    });

    doc.add_decision_point(DecisionPoint {
        id: "stable".into(),
        label: "Stable".into(),
        description: Some("Blood pressure at target".into()),
        branches: vec![],
        is_initial: false,
        is_terminal: true,
        invariants: vec![],
        urgency: None,
    });

    doc.add_safety_constraint(SafetyConstraint {
        id: "ace_arb_no_combine".into(),
        description: "Do not combine ACE inhibitor and ARB".into(),
        guard: GuidelineGuard::And(vec![
            GuidelineGuard::MedicationActive {
                medication: "lisinopril".into(),
            },
            GuidelineGuard::MedicationActive {
                medication: "losartan".into(),
            },
        ]),
        severity: ConstraintSeverity::Critical,
        applies_to: vec![],
    });

    doc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_document() {
        let doc = GuidelineDocument::new("Test");
        assert_eq!(doc.metadata.title, "Test");
        assert!(doc.decision_points.is_empty());
        assert!(!doc.id.is_empty());
    }

    #[test]
    fn test_dose_spec_display() {
        let d = DoseSpec::new(500.0, "mg").with_frequency("BID");
        assert_eq!(format!("{}", d), "500 mg BID");
    }

    #[test]
    fn test_dose_in_range() {
        let d = DoseSpec::new(50.0, "mg").with_range(10.0, 100.0);
        assert!(d.is_in_range());
        let d2 = DoseSpec::new(150.0, "mg").with_range(10.0, 100.0);
        assert!(!d2.is_in_range());
    }

    #[test]
    fn test_taper_schedule() {
        let ts = TaperSchedule {
            steps: vec![
                TaperStep { dose: DoseSpec::new(40.0, "mg"), duration_days: 7 },
                TaperStep { dose: DoseSpec::new(20.0, "mg"), duration_days: 7 },
                TaperStep { dose: DoseSpec::new(10.0, "mg"), duration_days: 7 },
            ],
        };
        assert_eq!(ts.total_days(), 21);
        assert!(ts.is_monotonic_decreasing());
    }

    #[test]
    fn test_guard_and_or() {
        let a = GuidelineGuard::True;
        let b = GuidelineGuard::ClinicalPredicate {
            parameter: "x".into(),
            operator: ComparisonOp::Gt,
            threshold: 1.0,
            unit: None,
        };
        let combined = a.and(b.clone());
        // True AND b => b
        assert_eq!(combined, b);

        let c = GuidelineGuard::False;
        let combined2 = c.or(b.clone());
        // False OR b => b
        assert_eq!(combined2, b);
    }

    #[test]
    fn test_guard_negate() {
        assert_eq!(GuidelineGuard::True.negate(), GuidelineGuard::False);
        assert_eq!(GuidelineGuard::False.negate(), GuidelineGuard::True);
        let g = GuidelineGuard::ClinicalPredicate {
            parameter: "x".into(),
            operator: ComparisonOp::Lt,
            threshold: 5.0,
            unit: None,
        };
        let neg = g.clone().negate();
        if let GuidelineGuard::Not(inner) = neg {
            assert_eq!(*inner, g);
        } else {
            panic!("Expected Not");
        }
    }

    #[test]
    fn test_guard_depth_and_leaf_count() {
        let g = GuidelineGuard::And(vec![
            GuidelineGuard::ClinicalPredicate {
                parameter: "a".into(),
                operator: ComparisonOp::Gt,
                threshold: 1.0,
                unit: None,
            },
            GuidelineGuard::Or(vec![
                GuidelineGuard::LabThreshold {
                    test_name: "b".into(),
                    operator: ComparisonOp::Lt,
                    value: 2.0,
                    unit: None,
                },
                GuidelineGuard::True,
            ]),
        ]);
        assert_eq!(g.depth(), 3);
        assert_eq!(g.leaf_count(), 2);
    }

    #[test]
    fn test_comparison_op() {
        assert!(ComparisonOp::Lt.evaluate(1.0, 2.0));
        assert!(!ComparisonOp::Lt.evaluate(2.0, 1.0));
        assert_eq!(ComparisonOp::Lt.flip(), ComparisonOp::Gt);
        assert_eq!(ComparisonOp::Lt.negate(), ComparisonOp::Ge);
    }

    #[test]
    fn test_diabetes_template() {
        let doc = standard_diabetes_template();
        assert!(doc.num_decision_points() >= 6);
        assert!(!doc.initial_points().is_empty());
        assert!(!doc.terminal_points().is_empty());
        let meds = doc.all_medications();
        assert!(meds.contains(&"metformin".to_string()));
    }

    #[test]
    fn test_hypertension_template() {
        let doc = standard_hypertension_template();
        assert!(doc.num_decision_points() >= 4);
        let meds = doc.all_medications();
        assert!(meds.contains(&"lisinopril".to_string()));
    }

    #[test]
    fn test_document_adjacency() {
        let doc = standard_diabetes_template();
        let adj = doc.adjacency();
        assert!(adj.contains_key("initial_assessment"));
        let targets = &adj["initial_assessment"];
        assert!(targets.contains(&"metformin_monotherapy".to_string()));
    }

    #[test]
    fn test_action_summary() {
        let a = GuidelineAction::StartMedication {
            medication: "aspirin".into(),
            dose: DoseSpec::new(81.0, "mg"),
            route: "oral".into(),
            reason: None,
        };
        assert!(a.summary().contains("aspirin"));
        assert!(a.is_medication_action());
        assert!(!a.is_monitoring_action());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let doc = standard_diabetes_template();
        let json = serde_json::to_string(&doc).unwrap();
        let doc2: GuidelineDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(doc2.metadata.title, doc.metadata.title);
        assert_eq!(doc2.decision_points.len(), doc.decision_points.len());
    }

    #[test]
    fn test_guard_rename_parameter() {
        let mut g = GuidelineGuard::And(vec![
            GuidelineGuard::ClinicalPredicate {
                parameter: "old_name".into(),
                operator: ComparisonOp::Gt,
                threshold: 1.0,
                unit: None,
            },
            GuidelineGuard::LabThreshold {
                test_name: "other".into(),
                operator: ComparisonOp::Lt,
                value: 5.0,
                unit: None,
            },
        ]);
        g.rename_parameter("old_name", "new_name");
        let params = g.referenced_parameters();
        assert!(params.contains(&"new_name".to_string()));
        assert!(!params.contains(&"old_name".to_string()));
    }

    #[test]
    fn test_evidence_level_display() {
        assert_eq!(EvidenceLevel::High.to_string(), "High");
        assert_eq!(EvidenceLevel::ExpertOpinion.to_string(), "ExpertOpinion");
    }
}

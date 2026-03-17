//! Safety verification types for the GuardPharma polypharmacy verification engine.
//!
//! This module defines the core safety types used throughout the verification pipeline,
//! including severity classifications, safety properties (with boolean combinators),
//! verification verdicts, tiered verification results, safety certificates, execution
//! traces with counterexamples, and aggregate safety reports.
//!
//! # Design Notes
//!
//! - [`SafetyProperty`] supports recursive boolean structure (conjunction, disjunction,
//!   negation) to express complex clinical constraints compositionally.
//! - [`SafetyVerdict`] forms a four-valued lattice ordered from [`SafetyVerdict::Unsafe`]
//!   (worst) to [`SafetyVerdict::Safe`] (best); merging always takes the worst case.
//! - [`CounterExample`] provides a full execution trace with a pinpointed violation step
//!   and a clinical narrative suitable for human review.
//! - All types are `Serialize`/`Deserialize` for JSON persistence and transport.

use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// 1. ConflictSeverity
// ---------------------------------------------------------------------------

/// Severity level for a drug-drug interaction conflict.
///
/// Ordered from most severe ([`Critical`](ConflictSeverity::Critical)) to least
/// ([`Minor`](ConflictSeverity::Minor)). The [`Ord`] implementation reflects this
/// ordering so that `Critical > Major > Moderate > Minor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ConflictSeverity {
    /// Life-threatening interaction requiring immediate clinical intervention.
    Critical,
    /// Serious interaction that may cause significant harm or treatment failure.
    Major,
    /// Clinically notable interaction that should be monitored.
    Moderate,
    /// Low-risk interaction unlikely to require dosage adjustment.
    Minor,
}

impl ConflictSeverity {
    /// Returns a numeric score where higher values indicate greater severity.
    ///
    /// | Variant    | Score |
    /// |------------|-------|
    /// | `Critical` |   4   |
    /// | `Major`    |   3   |
    /// | `Moderate` |   2   |
    /// | `Minor`    |   1   |
    pub fn numeric_score(self) -> u32 {
        match self {
            Self::Critical => 4,
            Self::Major => 3,
            Self::Moderate => 2,
            Self::Minor => 1,
        }
    }

    /// Returns `true` when the severity warrants immediate clinical action.
    ///
    /// Currently this is `true` for [`Critical`](Self::Critical) and
    /// [`Major`](Self::Major) interactions.
    pub fn requires_immediate_action(self) -> bool {
        matches!(self, Self::Critical | Self::Major)
    }

    /// Returns a CSS-style hex colour code suitable for UI rendering.
    ///
    /// | Variant    | Colour  |
    /// |------------|---------|
    /// | `Critical` | `#FF0000` (red)    |
    /// | `Major`    | `#FF8C00` (orange) |
    /// | `Moderate` | `#FFD700` (gold)   |
    /// | `Minor`    | `#32CD32` (green)  |
    pub fn color_code(self) -> &'static str {
        match self {
            Self::Critical => "#FF0000",
            Self::Major => "#FF8C00",
            Self::Moderate => "#FFD700",
            Self::Minor => "#32CD32",
        }
    }
}

impl PartialOrd for ConflictSeverity {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ConflictSeverity {
    /// `Critical > Major > Moderate > Minor`
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.numeric_score().cmp(&other.numeric_score())
    }
}

impl fmt::Display for ConflictSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::Major => write!(f, "Major"),
            Self::Moderate => write!(f, "Moderate"),
            Self::Minor => write!(f, "Minor"),
        }
    }
}

impl FromStr for ConflictSeverity {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "critical" => Ok(Self::Critical),
            "major" => Ok(Self::Major),
            "moderate" => Ok(Self::Moderate),
            "minor" => Ok(Self::Minor),
            other => Err(format!("unknown conflict severity: `{other}`")),
        }
    }
}

// ---------------------------------------------------------------------------
// 2. SafetyProperty
// ---------------------------------------------------------------------------

/// A declarative safety property that the verification engine checks.
///
/// Atomic variants describe concrete pharmacokinetic or clinical constraints.
/// The boolean combinators [`Conjunction`](Self::Conjunction),
/// [`Disjunction`](Self::Disjunction), and [`Negation`](Self::Negation) allow
/// composing arbitrarily complex properties from simpler ones.
///
/// # Examples
///
/// ```rust,ignore
/// let bound = SafetyProperty::ConcentrationBound {
///     drug_id: "warfarin".into(),
///     min: 1.0,
///     max: 4.0,
/// };
/// assert!(bound.is_atomic());
/// assert_eq!(bound.involves_drug(), vec!["warfarin".to_string()]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SafetyProperty {
    /// The steady-state plasma concentration of a drug must lie within `[min, max]` mg/L.
    ConcentrationBound {
        /// Drug identifier (e.g. `"warfarin"`).
        drug_id: String,
        /// Minimum acceptable concentration (mg/L).
        min: f64,
        /// Maximum acceptable concentration (mg/L).
        max: f64,
    },

    /// The combined severity of the interaction between two drugs must not exceed
    /// `max_severity`.
    InteractionLimit {
        /// First drug in the interaction pair.
        drug_a: String,
        /// Second drug in the interaction pair.
        drug_b: String,
        /// Highest tolerable severity level.
        max_severity: ConflictSeverity,
    },

    /// A generic boolean predicate keyed by `predicate_key` that encodes a
    /// clinical constraint described in natural language by `description`.
    ClinicalConstraint {
        /// Human-readable description of the clinical constraint.
        description: String,
        /// Machine-readable key used to look up the predicate evaluator.
        predicate_key: String,
    },

    /// A temporal constraint on the minimum or maximum duration (hours) between
    /// relevant clinical events.
    Temporal {
        /// Human-readable description of the temporal constraint.
        description: String,
        /// Minimum elapsed hours (inclusive) — `None` means no lower bound.
        min_hours: Option<f64>,
        /// Maximum elapsed hours (inclusive) — `None` means no upper bound.
        max_hours: Option<f64>,
    },

    /// All contained sub-properties must hold simultaneously (logical AND).
    Conjunction(Vec<SafetyProperty>),

    /// At least one contained sub-property must hold (logical OR).
    Disjunction(Vec<SafetyProperty>),

    /// The contained sub-property must **not** hold (logical NOT).
    Negation(Box<SafetyProperty>),
}

impl SafetyProperty {
    /// Returns a human-readable description of this property.
    pub fn description(&self) -> String {
        match self {
            Self::ConcentrationBound { drug_id, min, max } => {
                format!(
                    "Concentration of {drug_id} must be within [{min:.2}, {max:.2}] mg/L"
                )
            }
            Self::InteractionLimit {
                drug_a,
                drug_b,
                max_severity,
            } => {
                format!(
                    "Interaction between {drug_a} and {drug_b} must not exceed {max_severity} severity"
                )
            }
            Self::ClinicalConstraint { description, .. } => description.clone(),
            Self::Temporal {
                description,
                min_hours,
                max_hours,
            } => {
                let bounds = match (min_hours, max_hours) {
                    (Some(lo), Some(hi)) => format!(" [{lo:.1}h, {hi:.1}h]"),
                    (Some(lo), None) => format!(" [>={lo:.1}h]"),
                    (None, Some(hi)) => format!(" [<={hi:.1}h]"),
                    (None, None) => String::new(),
                };
                format!("{description}{bounds}")
            }
            Self::Conjunction(parts) => {
                let descs: Vec<_> = parts.iter().map(|p| p.description()).collect();
                format!("ALL OF ({})", descs.join(" AND "))
            }
            Self::Disjunction(parts) => {
                let descs: Vec<_> = parts.iter().map(|p| p.description()).collect();
                format!("ANY OF ({})", descs.join(" OR "))
            }
            Self::Negation(inner) => {
                format!("NOT ({})", inner.description())
            }
        }
    }

    /// Collects the identifiers of all drugs referenced by this property,
    /// recursing into boolean combinators. The returned list may contain
    /// duplicates when the same drug appears in multiple sub-properties.
    pub fn involves_drug(&self) -> Vec<String> {
        match self {
            Self::ConcentrationBound { drug_id, .. } => vec![drug_id.clone()],
            Self::InteractionLimit { drug_a, drug_b, .. } => {
                vec![drug_a.clone(), drug_b.clone()]
            }
            Self::ClinicalConstraint { .. } | Self::Temporal { .. } => vec![],
            Self::Conjunction(parts) | Self::Disjunction(parts) => {
                parts.iter().flat_map(|p| p.involves_drug()).collect()
            }
            Self::Negation(inner) => inner.involves_drug(),
        }
    }

    /// Returns `true` when this property is an atomic (leaf) constraint, i.e.
    /// not a boolean combinator.
    pub fn is_atomic(&self) -> bool {
        !matches!(
            self,
            Self::Conjunction(_) | Self::Disjunction(_) | Self::Negation(_)
        )
    }

    /// Produces a flat list of conjuncts.
    ///
    /// If `self` is a [`Conjunction`](Self::Conjunction), its children are
    /// returned (recursively flattened). Otherwise a singleton slice containing
    /// `self` is returned.
    pub fn conjuncts(&self) -> Vec<&SafetyProperty> {
        match self {
            Self::Conjunction(parts) => {
                parts.iter().flat_map(|p| p.conjuncts()).collect()
            }
            other => vec![other],
        }
    }

    /// Counts the number of atomic sub-properties reachable from this node.
    ///
    /// Boolean combinators add no complexity of their own; only leaf
    /// constraints contribute to the count.
    pub fn complexity(&self) -> usize {
        match self {
            Self::Conjunction(parts) | Self::Disjunction(parts) => {
                parts.iter().map(|p| p.complexity()).sum()
            }
            Self::Negation(inner) => inner.complexity(),
            _ => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// 3. SafetyVerdict
// ---------------------------------------------------------------------------

/// Four-valued verdict produced by the verification engine.
///
/// The values form a total order from safest to least safe:
///
/// ```text
/// Safe > PossiblySafe > PossiblyUnsafe > Unsafe
/// ```
///
/// The [`Ord`] implementation encodes `Safe` as the **greatest** element so
/// that `max()` yields the most optimistic verdict, while the [`merge`]
/// method takes the **worst-case** (minimum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum SafetyVerdict {
    /// The property is definitively satisfied.
    Safe,
    /// The property is likely satisfied but not fully proven.
    PossiblySafe,
    /// The property may be violated; further analysis is recommended.
    PossiblyUnsafe,
    /// The property is definitively violated.
    Unsafe,
}

impl SafetyVerdict {
    /// Numeric rank where **lower** values are **worse** (less safe).
    fn rank(self) -> u8 {
        match self {
            Self::Unsafe => 0,
            Self::PossiblyUnsafe => 1,
            Self::PossiblySafe => 2,
            Self::Safe => 3,
        }
    }

    fn from_rank(r: u8) -> Self {
        match r {
            0 => Self::Unsafe,
            1 => Self::PossiblyUnsafe,
            2 => Self::PossiblySafe,
            _ => Self::Safe,
        }
    }

    /// Returns `true` for the two definitive verdicts (`Safe` and `Unsafe`).
    pub fn is_definitive(self) -> bool {
        matches!(self, Self::Safe | Self::Unsafe)
    }

    /// Returns `true` when the verdict is at least possibly safe.
    pub fn is_safe_or_possibly_safe(self) -> bool {
        matches!(self, Self::Safe | Self::PossiblySafe)
    }

    /// Merges two verdicts by taking the **worst-case** (least safe) outcome.
    ///
    /// ```rust,ignore
    /// assert_eq!(SafetyVerdict::Safe.merge(&SafetyVerdict::Unsafe), SafetyVerdict::Unsafe);
    /// ```
    pub fn merge(self, other: &SafetyVerdict) -> SafetyVerdict {
        if self.rank() <= other.rank() {
            self
        } else {
            *other
        }
    }

    /// Shifts one step towards [`Safe`](Self::Safe). Already-safe verdicts are
    /// unchanged.
    pub fn upgrade(self) -> SafetyVerdict {
        Self::from_rank(self.rank().saturating_add(1).min(3))
    }

    /// Shifts one step towards [`Unsafe`](Self::Unsafe). Already-unsafe
    /// verdicts are unchanged.
    pub fn downgrade(self) -> SafetyVerdict {
        Self::from_rank(self.rank().saturating_sub(1))
    }
}

impl PartialOrd for SafetyVerdict {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SafetyVerdict {
    /// `Safe > PossiblySafe > PossiblyUnsafe > Unsafe`
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

impl fmt::Display for SafetyVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safe => write!(f, "Safe"),
            Self::PossiblySafe => write!(f, "Possibly Safe"),
            Self::PossiblyUnsafe => write!(f, "Possibly Unsafe"),
            Self::Unsafe => write!(f, "Unsafe"),
        }
    }
}

// ---------------------------------------------------------------------------
// 4. VerificationTier
// ---------------------------------------------------------------------------

/// The algorithmic tier used to produce a verification result.
///
/// The system implements a tiered verification architecture:
///
/// - **Tier 1** uses abstract interpretation for fast, sound
///   over-approximations.
/// - **Tier 2** uses explicit-state or bounded model checking for precise
///   verdicts when Tier 1 is inconclusive.
/// - **Combined** merges evidence from both tiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum VerificationTier {
    /// Abstract interpretation (fast, may over-approximate).
    #[serde(rename = "tier1_abstract_interpretation")]
    Tier1AbstractInterpretation,
    /// Model checking (precise, potentially expensive).
    #[serde(rename = "tier2_model_checking")]
    Tier2ModelChecking,
    /// Evidence combined from both tiers.
    #[serde(rename = "combined")]
    Combined,
}

impl fmt::Display for VerificationTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tier1AbstractInterpretation => {
                write!(f, "Tier 1 (Abstract Interpretation)")
            }
            Self::Tier2ModelChecking => write!(f, "Tier 2 (Model Checking)"),
            Self::Combined => write!(f, "Combined (Tier 1 + Tier 2)"),
        }
    }
}

// ---------------------------------------------------------------------------
// 5. VerificationResult
// ---------------------------------------------------------------------------

/// The outcome of verifying a single [`SafetyProperty`].
///
/// Produced by the verification engine for each property in a run. Contains
/// the verdict, supporting evidence strings, timing information, an optional
/// counterexample, and metadata linking back to the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// The four-valued safety verdict.
    pub verdict: SafetyVerdict,
    /// The property that was checked.
    pub property: SafetyProperty,
    /// Human-readable evidence strings supporting the verdict (may be empty).
    pub evidence: Vec<String>,
    /// Wall-clock duration of the verification in milliseconds.
    pub duration_ms: u64,
    /// Which algorithmic tier produced this result.
    pub tier: VerificationTier,
    /// An optional concrete counterexample witnessing a violation.
    pub counterexample: Option<CounterExample>,
    /// Identifier of the verification run that produced this result.
    pub run_id: Option<String>,
    /// UTC timestamp at which the result was generated.
    pub timestamp: DateTime<Utc>,
}

impl VerificationResult {
    /// Creates a new `VerificationResult` with the given core fields.
    ///
    /// The `timestamp` is set to [`Utc::now()`], `counterexample` and
    /// `run_id` default to `None`, and `evidence` starts empty.
    pub fn new(
        verdict: SafetyVerdict,
        property: SafetyProperty,
        tier: VerificationTier,
        duration_ms: u64,
    ) -> Self {
        Self {
            verdict,
            property,
            evidence: Vec::new(),
            duration_ms,
            tier,
            counterexample: None,
            run_id: None,
            timestamp: Utc::now(),
        }
    }

    /// Convenience: is the verdict [`Safe`](SafetyVerdict::Safe)?
    pub fn is_safe(&self) -> bool {
        self.verdict == SafetyVerdict::Safe
    }

    /// Returns `true` when a counterexample trace is attached.
    pub fn has_counterexample(&self) -> bool {
        self.counterexample.is_some()
    }

    /// One-line summary suitable for log output.
    pub fn summary(&self) -> String {
        let ce_marker = if self.has_counterexample() {
            " [counterexample attached]"
        } else {
            ""
        };
        format!(
            "[{}] {} — {} ({}ms, {}){}",
            self.verdict,
            self.property.description(),
            if self.evidence.is_empty() {
                "no evidence".to_string()
            } else {
                format!("{} evidence item(s)", self.evidence.len())
            },
            self.duration_ms,
            self.tier,
            ce_marker,
        )
    }
}

// ---------------------------------------------------------------------------
// 6. SafetyCertificate
// ---------------------------------------------------------------------------

/// A signed attestation that a [`SafetyProperty`] holds (or does not) for a
/// particular patient/drug configuration.
///
/// Certificates may carry an expiry time after which they should be
/// re-verified. The `confidence` field captures the statistical or analytical
/// confidence in the verdict (range `[0.0, 1.0]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyCertificate {
    /// Unique certificate identifier.
    pub id: String,
    /// The safety property that this certificate attests to.
    pub property: SafetyProperty,
    /// The verdict for the property.
    pub verdict: SafetyVerdict,
    /// When the certificate was issued.
    pub timestamp: DateTime<Utc>,
    /// Free-text description of the methodology (e.g. "abstract interpretation + BMC").
    pub methodology: String,
    /// The verification tier that produced the evidence.
    pub tier: VerificationTier,
    /// Confidence in the verdict, in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Optional expiry time after which the certificate should be re-verified.
    pub valid_until: Option<DateTime<Utc>>,
    /// Identity of the issuing system or authority.
    pub issuer: String,
}

impl SafetyCertificate {
    /// Creates a new certificate. `confidence` is clamped to `[0.0, 1.0]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        property: SafetyProperty,
        verdict: SafetyVerdict,
        methodology: String,
        tier: VerificationTier,
        confidence: f64,
        valid_until: Option<DateTime<Utc>>,
        issuer: String,
    ) -> Self {
        Self {
            id,
            property,
            verdict,
            timestamp: Utc::now(),
            methodology,
            tier,
            confidence: confidence.clamp(0.0, 1.0),
            valid_until,
            issuer,
        }
    }

    /// A certificate is considered **valid** when:
    /// 1. It is not expired, and
    /// 2. The verdict is at least [`PossiblySafe`](SafetyVerdict::PossiblySafe).
    pub fn is_valid(&self) -> bool {
        !self.is_expired() && self.verdict.is_safe_or_possibly_safe()
    }

    /// Returns `true` when `valid_until` is in the past.
    ///
    /// Certificates with no expiry (`valid_until == None`) never expire.
    pub fn is_expired(&self) -> bool {
        match self.valid_until {
            Some(expiry) => Utc::now() > expiry,
            None => false,
        }
    }

    /// Hours of remaining validity, or `None` if the certificate has no
    /// expiry or is already expired.
    pub fn remaining_validity_hours(&self) -> Option<f64> {
        self.valid_until.and_then(|expiry| {
            let remaining = expiry - Utc::now();
            let hours = remaining.num_milliseconds() as f64 / 3_600_000.0;
            if hours > 0.0 {
                Some(hours)
            } else {
                None
            }
        })
    }
}

// ---------------------------------------------------------------------------
// 7. TraceStep
// ---------------------------------------------------------------------------

/// A single step in an execution trace of the pharmacokinetic transition
/// system.
///
/// Each step records the simulation time, the PTA location, per-drug
/// concentrations, boolean clinical flags, and an optional action or note.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceStep {
    /// Simulation time in hours since dose administration.
    pub time: f64,
    /// Numeric identifier of the PTA location at this step.
    pub location_id: u32,
    /// Human-readable name of the PTA location (e.g. `"absorption_phase"`).
    pub location_name: String,
    /// Plasma concentration for each drug at this time step (mg/L).
    pub concentrations: IndexMap<String, f64>,
    /// Boolean clinical flags active at this time step.
    pub clinical_flags: IndexMap<String, bool>,
    /// An optional action label (e.g. `"administer_dose"`).
    pub action: Option<String>,
    /// An optional human-readable annotation.
    pub note: Option<String>,
}

impl TraceStep {
    /// Creates a `TraceStep` with the required positional fields; optional
    /// fields default to empty/`None`.
    pub fn new(
        time: f64,
        location_id: u32,
        location_name: String,
        concentrations: IndexMap<String, f64>,
    ) -> Self {
        Self {
            time,
            location_id,
            location_name,
            concentrations,
            clinical_flags: IndexMap::new(),
            action: None,
            note: None,
        }
    }

    /// Returns `true` when an action label is present.
    pub fn has_action(&self) -> bool {
        self.action.is_some()
    }

    /// Looks up the concentration of a specific drug at this time step.
    pub fn concentration_of(&self, drug_id: &str) -> Option<f64> {
        self.concentrations.get(drug_id).copied()
    }
}

// ---------------------------------------------------------------------------
// 8. CounterExample
// ---------------------------------------------------------------------------

/// A concrete counterexample demonstrating how a safety property is violated.
///
/// Consists of a full execution [`trace`](Self::trace), a
/// [`violation_point`](Self::violation_point) index pinpointing the step at
/// which the violation occurs, and a [`clinical_narrative`](Self::clinical_narrative)
/// explaining the scenario in clinical terms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CounterExample {
    /// The full execution trace from initial state through the violation.
    pub trace: Vec<TraceStep>,
    /// Zero-based index into `trace` where the violation is first observed.
    pub violation_point: usize,
    /// Short machine-oriented description of the violation.
    pub violation_description: String,
    /// Longer, clinician-friendly narrative of the scenario.
    pub clinical_narrative: String,
    /// Drugs involved in the violation.
    pub involved_drugs: Vec<String>,
    /// Severity of the demonstrated conflict.
    pub severity: ConflictSeverity,
}

impl CounterExample {
    /// Creates a new counterexample. Panics if `violation_point >= trace.len()`
    /// in debug builds (returns gracefully in release).
    pub fn new(
        trace: Vec<TraceStep>,
        violation_point: usize,
        violation_description: String,
        clinical_narrative: String,
        involved_drugs: Vec<String>,
        severity: ConflictSeverity,
    ) -> Self {
        debug_assert!(
            violation_point < trace.len() || trace.is_empty(),
            "violation_point ({violation_point}) out of bounds for trace of length {}",
            trace.len()
        );
        Self {
            trace,
            violation_point,
            violation_description,
            clinical_narrative,
            involved_drugs,
            severity,
        }
    }

    /// Returns the [`TraceStep`] where the violation occurs, or `None` if the
    /// index is out of bounds.
    pub fn violation_step(&self) -> Option<&TraceStep> {
        self.trace.get(self.violation_point)
    }

    /// Total number of steps in the trace.
    pub fn trace_length(&self) -> usize {
        self.trace.len()
    }

    /// One-line summary for logging / CLI display.
    pub fn summary(&self) -> String {
        let drugs = self.involved_drugs.join(", ");
        let time_info = self
            .violation_step()
            .map(|s| format!(" at t={:.2}h", s.time))
            .unwrap_or_default();
        format!(
            "[{severity}] {desc}{time_info} — drugs: [{drugs}] ({steps} trace step(s))",
            severity = self.severity,
            desc = self.violation_description,
            steps = self.trace.len(),
        )
    }

    /// Returns the prefix of the trace **before** the violation step
    /// (exclusive).
    pub fn trace_before_violation(&self) -> &[TraceStep] {
        if self.violation_point > self.trace.len() {
            &self.trace[..]
        } else {
            &self.trace[..self.violation_point]
        }
    }

    /// Returns the suffix of the trace starting at (and including) the
    /// violation step.
    pub fn trace_at_and_after_violation(&self) -> &[TraceStep] {
        if self.violation_point >= self.trace.len() {
            &[]
        } else {
            &self.trace[self.violation_point..]
        }
    }
}

// ---------------------------------------------------------------------------
// 9. SafetyReport
// ---------------------------------------------------------------------------

/// An aggregate safety report for a single patient or verification run.
///
/// Collects all [`VerificationResult`]s and [`SafetyCertificate`]s and
/// provides convenience queries over the set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyReport {
    /// Patient or scenario identifier.
    pub patient_id: String,
    /// Identifier of the verification run.
    pub run_id: String,
    /// Individual verification results.
    pub results: Vec<VerificationResult>,
    /// Certificates issued for this run.
    pub certificates: Vec<SafetyCertificate>,
    /// Pre-computed overall verdict (worst-case merge of all results).
    pub overall_verdict: SafetyVerdict,
    /// When the report was generated.
    pub generated_at: DateTime<Utc>,
    /// Free-text summary of the engine configuration used.
    pub configuration_summary: String,
}

impl SafetyReport {
    /// Creates a new report, automatically computing the
    /// [`overall_verdict`](Self::overall_verdict) from the supplied results.
    pub fn new(
        patient_id: String,
        run_id: String,
        results: Vec<VerificationResult>,
        certificates: Vec<SafetyCertificate>,
        configuration_summary: String,
    ) -> Self {
        let overall_verdict = results
            .iter()
            .map(|r| r.verdict)
            .fold(SafetyVerdict::Safe, |acc, v| acc.merge(&v));
        Self {
            patient_id,
            run_id,
            results,
            certificates,
            overall_verdict,
            generated_at: Utc::now(),
            configuration_summary,
        }
    }

    /// Recomputes and returns the worst-case verdict across all results.
    ///
    /// Returns [`Safe`](SafetyVerdict::Safe) when the result set is empty.
    pub fn worst_verdict(&self) -> SafetyVerdict {
        self.results
            .iter()
            .map(|r| r.verdict)
            .fold(SafetyVerdict::Safe, |acc, v| acc.merge(&v))
    }

    /// Number of results with an [`Unsafe`](SafetyVerdict::Unsafe) verdict.
    pub fn num_unsafe(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.verdict == SafetyVerdict::Unsafe)
            .count()
    }

    /// Number of results with a [`Safe`](SafetyVerdict::Safe) verdict.
    pub fn num_safe(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.verdict == SafetyVerdict::Safe)
            .count()
    }

    /// Returns `true` when at least one result carries a counterexample.
    pub fn has_counterexamples(&self) -> bool {
        self.results.iter().any(|r| r.has_counterexample())
    }

    /// Collects references to every counterexample across all results.
    pub fn all_counterexamples(&self) -> Vec<&CounterExample> {
        self.results
            .iter()
            .filter_map(|r| r.counterexample.as_ref())
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    // -- helpers -----------------------------------------------------------

    /// Build a minimal trace with `n` steps, each 1 hour apart.
    fn make_trace(n: usize, drug: &str) -> Vec<TraceStep> {
        (0..n)
            .map(|i| {
                let mut concs = IndexMap::new();
                concs.insert(drug.to_string(), 1.0 + i as f64);
                TraceStep::new(i as f64, i as u32, format!("loc_{i}"), concs)
            })
            .collect()
    }

    fn sample_property() -> SafetyProperty {
        SafetyProperty::ConcentrationBound {
            drug_id: "warfarin".into(),
            min: 1.0,
            max: 4.0,
        }
    }

    // -- 1. Severity ordering & scoring ------------------------------------

    #[test]
    fn severity_ordering_and_score() {
        assert!(ConflictSeverity::Critical > ConflictSeverity::Major);
        assert!(ConflictSeverity::Major > ConflictSeverity::Moderate);
        assert!(ConflictSeverity::Moderate > ConflictSeverity::Minor);

        assert_eq!(ConflictSeverity::Critical.numeric_score(), 4);
        assert_eq!(ConflictSeverity::Minor.numeric_score(), 1);
    }

    #[test]
    fn severity_immediate_action() {
        assert!(ConflictSeverity::Critical.requires_immediate_action());
        assert!(ConflictSeverity::Major.requires_immediate_action());
        assert!(!ConflictSeverity::Moderate.requires_immediate_action());
        assert!(!ConflictSeverity::Minor.requires_immediate_action());
    }

    #[test]
    fn severity_display_and_from_str() {
        for sev in [
            ConflictSeverity::Critical,
            ConflictSeverity::Major,
            ConflictSeverity::Moderate,
            ConflictSeverity::Minor,
        ] {
            let s = sev.to_string();
            let parsed: ConflictSeverity = s.parse().unwrap();
            assert_eq!(parsed, sev);
        }
        assert!("unknown".parse::<ConflictSeverity>().is_err());
    }

    // -- 2. SafetyProperty construction & description ----------------------

    #[test]
    fn property_concentration_bound_description_and_drugs() {
        let prop = sample_property();
        assert!(prop.description().contains("warfarin"));
        assert!(prop.description().contains("1.00"));
        assert_eq!(prop.involves_drug(), vec!["warfarin".to_string()]);
        assert!(prop.is_atomic());
        assert_eq!(prop.complexity(), 1);
    }

    #[test]
    fn property_conjunction_flattening_and_complexity() {
        let a = SafetyProperty::ConcentrationBound {
            drug_id: "A".into(),
            min: 0.0,
            max: 5.0,
        };
        let b = SafetyProperty::InteractionLimit {
            drug_a: "A".into(),
            drug_b: "B".into(),
            max_severity: ConflictSeverity::Moderate,
        };
        let inner_conj = SafetyProperty::Conjunction(vec![a.clone(), b.clone()]);
        let outer_conj = SafetyProperty::Conjunction(vec![inner_conj, a.clone()]);

        // Flattened conjuncts should have 3 elements.
        assert_eq!(outer_conj.conjuncts().len(), 3);
        assert!(!outer_conj.is_atomic());
        assert_eq!(outer_conj.complexity(), 3);

        // involves_drug collects from all sub-properties.
        let drugs = outer_conj.involves_drug();
        assert!(drugs.contains(&"A".to_string()));
        assert!(drugs.contains(&"B".to_string()));
    }

    #[test]
    fn property_negation_and_disjunction() {
        let a = sample_property();
        let neg = SafetyProperty::Negation(Box::new(a.clone()));
        assert!(!neg.is_atomic());
        assert_eq!(neg.complexity(), 1);
        assert!(neg.description().starts_with("NOT"));

        let disj = SafetyProperty::Disjunction(vec![a.clone(), neg]);
        assert!(disj.description().starts_with("ANY OF"));
        assert_eq!(disj.complexity(), 2);
    }

    #[test]
    fn property_temporal_description() {
        let t = SafetyProperty::Temporal {
            description: "Wait after dose".into(),
            min_hours: Some(2.0),
            max_hours: Some(6.0),
        };
        let desc = t.description();
        assert!(desc.contains("2.0h"));
        assert!(desc.contains("6.0h"));

        let t_open = SafetyProperty::Temporal {
            description: "No upper bound".into(),
            min_hours: Some(1.0),
            max_hours: None,
        };
        assert!(t_open.description().contains(">=1.0h"));
    }

    // -- 3. Verdict ordering, merge, upgrade, downgrade --------------------

    #[test]
    fn verdict_ordering_and_merge() {
        assert!(SafetyVerdict::Safe > SafetyVerdict::PossiblySafe);
        assert!(SafetyVerdict::PossiblySafe > SafetyVerdict::PossiblyUnsafe);
        assert!(SafetyVerdict::PossiblyUnsafe > SafetyVerdict::Unsafe);

        // merge takes worst case
        assert_eq!(
            SafetyVerdict::Safe.merge(&SafetyVerdict::Unsafe),
            SafetyVerdict::Unsafe
        );
        assert_eq!(
            SafetyVerdict::PossiblySafe.merge(&SafetyVerdict::PossiblyUnsafe),
            SafetyVerdict::PossiblyUnsafe
        );
        assert_eq!(
            SafetyVerdict::Safe.merge(&SafetyVerdict::Safe),
            SafetyVerdict::Safe
        );
    }

    #[test]
    fn verdict_upgrade_downgrade() {
        assert_eq!(SafetyVerdict::Unsafe.upgrade(), SafetyVerdict::PossiblyUnsafe);
        assert_eq!(SafetyVerdict::Safe.upgrade(), SafetyVerdict::Safe);
        assert_eq!(SafetyVerdict::Safe.downgrade(), SafetyVerdict::PossiblySafe);
        assert_eq!(SafetyVerdict::Unsafe.downgrade(), SafetyVerdict::Unsafe);
    }

    #[test]
    fn verdict_predicates() {
        assert!(SafetyVerdict::Safe.is_definitive());
        assert!(SafetyVerdict::Unsafe.is_definitive());
        assert!(!SafetyVerdict::PossiblySafe.is_definitive());

        assert!(SafetyVerdict::Safe.is_safe_or_possibly_safe());
        assert!(SafetyVerdict::PossiblySafe.is_safe_or_possibly_safe());
        assert!(!SafetyVerdict::Unsafe.is_safe_or_possibly_safe());
    }

    // -- 4. CounterExample trace navigation --------------------------------

    #[test]
    fn counterexample_trace_navigation() {
        let trace = make_trace(5, "warfarin");
        let ce = CounterExample::new(
            trace,
            2,
            "concentration exceeded".into(),
            "At t=2h the plasma concentration of warfarin exceeded the therapeutic window.".into(),
            vec!["warfarin".into()],
            ConflictSeverity::Major,
        );

        assert_eq!(ce.trace_length(), 5);
        assert_eq!(ce.violation_step().unwrap().time, 2.0);
        assert_eq!(ce.trace_before_violation().len(), 2);
        assert_eq!(ce.trace_at_and_after_violation().len(), 3);

        let summary = ce.summary();
        assert!(summary.contains("Major"));
        assert!(summary.contains("warfarin"));
    }

    // -- 5. Certificate validity -------------------------------------------

    #[test]
    fn certificate_validity_and_expiry() {
        let future = Utc::now() + Duration::hours(24);
        let cert = SafetyCertificate::new(
            "cert-001".into(),
            sample_property(),
            SafetyVerdict::Safe,
            "abstract interpretation".into(),
            VerificationTier::Tier1AbstractInterpretation,
            0.95,
            Some(future),
            "guardpharma-engine".into(),
        );

        assert!(cert.is_valid());
        assert!(!cert.is_expired());
        let hours = cert.remaining_validity_hours().unwrap();
        assert!(hours > 23.0 && hours <= 24.0);

        // Expired certificate
        let past = Utc::now() - Duration::hours(1);
        let expired = SafetyCertificate::new(
            "cert-002".into(),
            sample_property(),
            SafetyVerdict::Safe,
            "model checking".into(),
            VerificationTier::Tier2ModelChecking,
            0.99,
            Some(past),
            "guardpharma-engine".into(),
        );
        assert!(expired.is_expired());
        assert!(!expired.is_valid());
        assert!(expired.remaining_validity_hours().is_none());
    }

    #[test]
    fn certificate_no_expiry() {
        let cert = SafetyCertificate::new(
            "cert-003".into(),
            sample_property(),
            SafetyVerdict::Safe,
            "combined".into(),
            VerificationTier::Combined,
            1.0,
            None,
            "test".into(),
        );
        assert!(!cert.is_expired());
        assert!(cert.remaining_validity_hours().is_none());
    }

    // -- 6. Report aggregation ---------------------------------------------

    #[test]
    fn report_aggregation() {
        let r_safe = VerificationResult::new(
            SafetyVerdict::Safe,
            sample_property(),
            VerificationTier::Tier1AbstractInterpretation,
            10,
        );

        let mut r_unsafe = VerificationResult::new(
            SafetyVerdict::Unsafe,
            SafetyProperty::InteractionLimit {
                drug_a: "warfarin".into(),
                drug_b: "aspirin".into(),
                max_severity: ConflictSeverity::Moderate,
            },
            VerificationTier::Tier2ModelChecking,
            250,
        );
        r_unsafe.counterexample = Some(CounterExample::new(
            make_trace(3, "warfarin"),
            1,
            "interaction exceeded limit".into(),
            "Clinical narrative here.".into(),
            vec!["warfarin".into(), "aspirin".into()],
            ConflictSeverity::Critical,
        ));

        let report = SafetyReport::new(
            "PT-12345".into(),
            "VR-run-1".into(),
            vec![r_safe, r_unsafe],
            vec![],
            "default config".into(),
        );

        assert_eq!(report.overall_verdict, SafetyVerdict::Unsafe);
        assert_eq!(report.worst_verdict(), SafetyVerdict::Unsafe);
        assert_eq!(report.num_safe(), 1);
        assert_eq!(report.num_unsafe(), 1);
        assert!(report.has_counterexamples());
        assert_eq!(report.all_counterexamples().len(), 1);
    }

    // -- 7. Serialization round-trips --------------------------------------

    #[test]
    fn severity_serde_roundtrip() {
        for sev in [
            ConflictSeverity::Critical,
            ConflictSeverity::Major,
            ConflictSeverity::Moderate,
            ConflictSeverity::Minor,
        ] {
            let json = serde_json::to_string(&sev).unwrap();
            let back: ConflictSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(back, sev);
        }
    }

    #[test]
    fn verdict_serde_roundtrip() {
        for v in [
            SafetyVerdict::Safe,
            SafetyVerdict::PossiblySafe,
            SafetyVerdict::PossiblyUnsafe,
            SafetyVerdict::Unsafe,
        ] {
            let json = serde_json::to_string(&v).unwrap();
            let back: SafetyVerdict = serde_json::from_str(&json).unwrap();
            assert_eq!(back, v);
        }
    }

    #[test]
    fn full_result_serde_roundtrip() {
        let mut result = VerificationResult::new(
            SafetyVerdict::PossiblyUnsafe,
            SafetyProperty::Conjunction(vec![
                sample_property(),
                SafetyProperty::Temporal {
                    description: "gap".into(),
                    min_hours: Some(4.0),
                    max_hours: None,
                },
            ]),
            VerificationTier::Combined,
            42,
        );
        result.evidence = vec!["evidence line 1".into(), "evidence line 2".into()];
        result.run_id = Some("VR-abc".into());
        result.counterexample = Some(CounterExample::new(
            make_trace(2, "metformin"),
            1,
            "temporal violation".into(),
            "The gap was too short.".into(),
            vec!["metformin".into()],
            ConflictSeverity::Moderate,
        ));

        let json = serde_json::to_string_pretty(&result).unwrap();
        let back: VerificationResult = serde_json::from_str(&json).unwrap();

        assert_eq!(back.verdict, result.verdict);
        assert_eq!(back.evidence.len(), 2);
        assert!(back.has_counterexample());
        assert_eq!(
            back.counterexample.as_ref().unwrap().trace_length(),
            2
        );
    }

    // -- 8. TraceStep concentration lookup ---------------------------------

    #[test]
    fn trace_step_concentration_lookup() {
        let mut concs = IndexMap::new();
        concs.insert("warfarin".to_string(), 2.5);
        concs.insert("aspirin".to_string(), 0.8);

        let step = TraceStep::new(1.0, 0, "absorption".into(), concs);
        assert_eq!(step.concentration_of("warfarin"), Some(2.5));
        assert_eq!(step.concentration_of("aspirin"), Some(0.8));
        assert_eq!(step.concentration_of("unknown"), None);
        assert!(!step.has_action());
    }

    // -- 9. VerificationResult summary format ------------------------------

    #[test]
    fn verification_result_summary_format() {
        let r = VerificationResult::new(
            SafetyVerdict::Safe,
            sample_property(),
            VerificationTier::Tier1AbstractInterpretation,
            7,
        );
        let s = r.summary();
        assert!(s.contains("Safe"));
        assert!(s.contains("7ms"));
        assert!(s.contains("Tier 1"));
        assert!(!s.contains("counterexample"));
    }

    // -- 10. Confidence clamping on certificate ----------------------------

    #[test]
    fn certificate_confidence_clamping() {
        let cert = SafetyCertificate::new(
            "cert-clamp".into(),
            sample_property(),
            SafetyVerdict::Safe,
            "test".into(),
            VerificationTier::Combined,
            1.5,
            None,
            "test".into(),
        );
        assert!((cert.confidence - 1.0).abs() < f64::EPSILON);

        let cert2 = SafetyCertificate::new(
            "cert-clamp2".into(),
            sample_property(),
            SafetyVerdict::Safe,
            "test".into(),
            VerificationTier::Combined,
            -0.5,
            None,
            "test".into(),
        );
        assert!((cert2.confidence - 0.0).abs() < f64::EPSILON);
    }

    // -- 11. Empty report defaults ----------------------------------------

    #[test]
    fn empty_report_defaults() {
        let report = SafetyReport::new(
            "PT-empty".into(),
            "VR-empty".into(),
            vec![],
            vec![],
            "none".into(),
        );
        assert_eq!(report.overall_verdict, SafetyVerdict::Safe);
        assert_eq!(report.worst_verdict(), SafetyVerdict::Safe);
        assert_eq!(report.num_safe(), 0);
        assert_eq!(report.num_unsafe(), 0);
        assert!(!report.has_counterexamples());
        assert!(report.all_counterexamples().is_empty());
    }

    // -- 12. Verification tier display ------------------------------------

    #[test]
    fn verification_tier_display_and_serde() {
        let tiers = [
            VerificationTier::Tier1AbstractInterpretation,
            VerificationTier::Tier2ModelChecking,
            VerificationTier::Combined,
        ];
        for tier in &tiers {
            let display = tier.to_string();
            assert!(!display.is_empty());

            let json = serde_json::to_string(tier).unwrap();
            let back: VerificationTier = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, tier);
        }
        assert!(tiers[0].to_string().contains("Abstract"));
        assert!(tiers[1].to_string().contains("Model Checking"));
        assert!(tiers[2].to_string().contains("Combined"));
    }
}

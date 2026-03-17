//! Cascade analysis types for modeling failure propagation in service topologies.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;
use std::ops::Deref;

use bitvec::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::service::ServiceId;

// ---------------------------------------------------------------------------
// FailureMode
// ---------------------------------------------------------------------------

/// Describes how a service has failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FailureMode {
    ServiceDown,
    HighLatency { latency_ms: u64 },
    PartialFailure { error_rate: f64 },
    ResourceExhaustion { utilization: f64 },
}

impl FailureMode {
    /// Severity score: ServiceDown=3, HighLatency=2, PartialFailure=1, ResourceExhaustion=2.
    pub fn severity_score(&self) -> u8 {
        match self {
            FailureMode::ServiceDown => 3,
            FailureMode::HighLatency { .. } => 2,
            FailureMode::PartialFailure { .. } => 1,
            FailureMode::ResourceExhaustion { .. } => 2,
        }
    }

    /// Returns `true` only for `ServiceDown`.
    pub fn is_total_failure(&self) -> bool {
        matches!(self, FailureMode::ServiceDown)
    }
}

impl fmt::Display for FailureMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FailureMode::ServiceDown => write!(f, "ServiceDown"),
            FailureMode::HighLatency { latency_ms } => {
                write!(f, "HighLatency({}ms)", latency_ms)
            }
            FailureMode::PartialFailure { error_rate } => {
                write!(f, "PartialFailure(err={:.2}%)", error_rate * 100.0)
            }
            FailureMode::ResourceExhaustion { utilization } => {
                write!(f, "ResourceExhaustion({:.1}%)", utilization * 100.0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FailureSet
// ---------------------------------------------------------------------------

/// Compact bitset representing a set of failed service indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FailureSet {
    bits: BitVec,
    cap: usize,
}

impl FailureSet {
    /// Create an empty failure set with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            bits: bitvec![0; capacity],
            cap: capacity,
        }
    }

    /// Create a failure set with the specified indices already set.
    pub fn with_failures(capacity: usize, indices: &[usize]) -> Self {
        let mut fs = Self::new(capacity);
        for &idx in indices {
            fs.insert(idx);
        }
        fs
    }

    /// Mark a service index as failed.
    pub fn insert(&mut self, index: usize) {
        if index < self.cap {
            self.bits.set(index, true);
        }
    }

    /// Mark a service index as no longer failed.
    pub fn remove(&mut self, index: usize) {
        if index < self.cap {
            self.bits.set(index, false);
        }
    }

    /// Returns `true` if the given index is in the failure set.
    pub fn contains(&self, index: usize) -> bool {
        if index < self.cap {
            self.bits[index]
        } else {
            false
        }
    }

    /// Number of set (failed) bits.
    pub fn len(&self) -> usize {
        self.bits.count_ones()
    }

    /// Returns `true` if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The total capacity of the bitset.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Iterate over the indices of set bits.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter_ones()
    }

    /// Set-union of two failure sets.
    pub fn union(&self, other: &FailureSet) -> FailureSet {
        let cap = self.cap.max(other.cap);
        let mut result = FailureSet::new(cap);
        for i in self.iter_set() {
            result.insert(i);
        }
        for i in other.iter_set() {
            result.insert(i);
        }
        result
    }

    /// Set-intersection of two failure sets.
    pub fn intersection(&self, other: &FailureSet) -> FailureSet {
        let cap = self.cap.min(other.cap);
        let mut result = FailureSet::new(cap);
        for i in 0..cap {
            if self.contains(i) && other.contains(i) {
                result.insert(i);
            }
        }
        result
    }

    /// Returns `true` if every set bit in `self` is also set in `other`.
    pub fn is_subset(&self, other: &FailureSet) -> bool {
        for i in self.iter_set() {
            if !other.contains(i) {
                return false;
            }
        }
        true
    }

    /// Returns `true` if every set bit in `other` is also set in `self`.
    pub fn is_superset(&self, other: &FailureSet) -> bool {
        other.is_subset(self)
    }

    fn to_indices(&self) -> Vec<usize> {
        self.iter_set().collect()
    }
}

impl Serialize for FailureSet {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Repr {
            capacity: usize,
            indices: Vec<usize>,
        }
        let repr = Repr {
            capacity: self.cap,
            indices: self.to_indices(),
        };
        repr.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FailureSet {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Repr {
            capacity: usize,
            indices: Vec<usize>,
        }
        let repr = Repr::deserialize(deserializer)?;
        Ok(FailureSet::with_failures(repr.capacity, &repr.indices))
    }
}

// ---------------------------------------------------------------------------
// PropagationStep
// ---------------------------------------------------------------------------

/// A single step in a failure-propagation simulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropagationStep {
    pub time_step: u32,
    pub service: ServiceId,
    pub load: f64,
    pub state: String,
    pub cause: Option<String>,
}

impl fmt::Display for PropagationStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "t={}: {} load={:.2} state={}",
            self.time_step, self.service, self.load, self.state
        )?;
        if let Some(ref cause) = self.cause {
            write!(f, " cause={}", cause)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PropagationTrace
// ---------------------------------------------------------------------------

/// Ordered sequence of propagation steps forming a simulation trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropagationTrace(Vec<PropagationStep>);

impl PropagationTrace {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn push(&mut self, step: PropagationStep) {
        self.0.push(step);
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn steps(&self) -> &[PropagationStep] {
        &self.0
    }

    /// Distinct set of services that appear in the trace.
    pub fn services_affected(&self) -> BTreeSet<ServiceId> {
        self.0.iter().map(|s| s.service.clone()).collect()
    }

    /// Highest load value observed across all steps.
    pub fn max_load(&self) -> f64 {
        self.0
            .iter()
            .map(|s| s.load)
            .fold(0.0_f64, f64::max)
    }

    /// First time-step at which any service enters the `"failed"` state.
    pub fn time_to_cascade(&self) -> Option<u32> {
        self.0
            .iter()
            .filter(|s| s.state == "failed")
            .map(|s| s.time_step)
            .min()
    }
}

impl Default for PropagationTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for PropagationTrace {
    type Target = [PropagationStep];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// CascadeSeverity
// ---------------------------------------------------------------------------

/// Severity classification for a cascade event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CascadeSeverity {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl CascadeSeverity {
    /// Numeric score: None=0, Low=1, Medium=2, High=3, Critical=4.
    pub fn score(&self) -> u8 {
        match self {
            CascadeSeverity::None => 0,
            CascadeSeverity::Low => 1,
            CascadeSeverity::Medium => 2,
            CascadeSeverity::High => 3,
            CascadeSeverity::Critical => 4,
        }
    }

    /// Build a severity from a 0-4 score (saturates at Critical).
    pub fn from_score(score: u8) -> Self {
        match score {
            0 => CascadeSeverity::None,
            1 => CascadeSeverity::Low,
            2 => CascadeSeverity::Medium,
            3 => CascadeSeverity::High,
            _ => CascadeSeverity::Critical,
        }
    }
}

impl PartialOrd for CascadeSeverity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CascadeSeverity {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score().cmp(&other.score())
    }
}

impl fmt::Display for CascadeSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            CascadeSeverity::None => "None",
            CascadeSeverity::Low => "Low",
            CascadeSeverity::Medium => "Medium",
            CascadeSeverity::High => "High",
            CascadeSeverity::Critical => "Critical",
        };
        write!(f, "{}", label)
    }
}

// ---------------------------------------------------------------------------
// CascadeClassification
// ---------------------------------------------------------------------------

/// Classification of the dominant cascade mechanism.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CascadeClassification {
    RetryAmplification,
    TimeoutChainViolation,
    FanInStorm,
    MultiFailure,
    Combined(Vec<CascadeClassification>),
}

impl CascadeClassification {
    pub fn description(&self) -> &str {
        match self {
            CascadeClassification::RetryAmplification => {
                "Cascade driven by retry amplification"
            }
            CascadeClassification::TimeoutChainViolation => {
                "Cascade driven by timeout chain violation"
            }
            CascadeClassification::FanInStorm => {
                "Cascade driven by fan-in overload storm"
            }
            CascadeClassification::MultiFailure => {
                "Cascade driven by multiple simultaneous failures"
            }
            CascadeClassification::Combined(_) => {
                "Cascade driven by a combination of mechanisms"
            }
        }
    }
}

impl fmt::Display for CascadeClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CascadeClassification::RetryAmplification => write!(f, "RetryAmplification"),
            CascadeClassification::TimeoutChainViolation => {
                write!(f, "TimeoutChainViolation")
            }
            CascadeClassification::FanInStorm => write!(f, "FanInStorm"),
            CascadeClassification::MultiFailure => write!(f, "MultiFailure"),
            CascadeClassification::Combined(inner) => {
                write!(f, "Combined(")?;
                for (i, c) in inner.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", c)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CascadeMetrics
// ---------------------------------------------------------------------------

/// Aggregate numeric metrics for a cascade analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CascadeMetrics {
    pub amplification_factor: f64,
    pub max_load: f64,
    pub affected_count: usize,
    pub time_to_cascade: Option<u32>,
    pub total_excess_load: f64,
    pub reliability_impact: f64,
}

impl CascadeMetrics {
    /// Derive a severity level from the metric values.
    pub fn severity(&self) -> CascadeSeverity {
        if self.amplification_factor >= 10.0 || self.reliability_impact >= 0.8 {
            CascadeSeverity::Critical
        } else if self.amplification_factor >= 5.0 || self.reliability_impact >= 0.5 {
            CascadeSeverity::High
        } else if self.amplification_factor >= 2.0 || self.reliability_impact >= 0.2 {
            CascadeSeverity::Medium
        } else if self.amplification_factor > 1.0 || self.reliability_impact > 0.0 {
            CascadeSeverity::Low
        } else {
            CascadeSeverity::None
        }
    }
}

// ---------------------------------------------------------------------------
// CascadeScenario
// ---------------------------------------------------------------------------

/// A single cascade scenario combining failure inputs with observed outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeScenario {
    pub id: String,
    pub failure_set: FailureSet,
    pub failure_modes: Vec<(usize, FailureMode)>,
    pub affected_services: Vec<ServiceId>,
    pub propagation_trace: PropagationTrace,
    pub severity: CascadeSeverity,
    pub classification: Option<CascadeClassification>,
    pub description: String,
}

/// Builder for [`CascadeScenario`].
#[derive(Debug, Default)]
pub struct CascadeScenarioBuilder {
    id: Option<String>,
    failure_set: Option<FailureSet>,
    failure_modes: Vec<(usize, FailureMode)>,
    affected_services: Vec<ServiceId>,
    propagation_trace: Option<PropagationTrace>,
    severity: Option<CascadeSeverity>,
    classification: Option<CascadeClassification>,
    description: Option<String>,
}

impl CascadeScenarioBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn failure_set(mut self, fs: FailureSet) -> Self {
        self.failure_set = Some(fs);
        self
    }

    pub fn failure_modes(mut self, modes: Vec<(usize, FailureMode)>) -> Self {
        self.failure_modes = modes;
        self
    }

    pub fn add_failure_mode(mut self, index: usize, mode: FailureMode) -> Self {
        self.failure_modes.push((index, mode));
        self
    }

    pub fn affected_services(mut self, services: Vec<ServiceId>) -> Self {
        self.affected_services = services;
        self
    }

    pub fn propagation_trace(mut self, trace: PropagationTrace) -> Self {
        self.propagation_trace = Some(trace);
        self
    }

    pub fn severity(mut self, severity: CascadeSeverity) -> Self {
        self.severity = Some(severity);
        self
    }

    pub fn classification(mut self, classification: CascadeClassification) -> Self {
        self.classification = Some(classification);
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn build(self) -> CascadeScenario {
        CascadeScenario {
            id: self.id.unwrap_or_default(),
            failure_set: self.failure_set.unwrap_or_else(|| FailureSet::new(0)),
            failure_modes: self.failure_modes,
            affected_services: self.affected_services,
            propagation_trace: self
                .propagation_trace
                .unwrap_or_else(PropagationTrace::new),
            severity: self.severity.unwrap_or(CascadeSeverity::None),
            classification: self.classification,
            description: self.description.unwrap_or_default(),
        }
    }
}

impl CascadeScenario {
    pub fn builder() -> CascadeScenarioBuilder {
        CascadeScenarioBuilder::new()
    }
}

// ---------------------------------------------------------------------------
// MinimalFailureSet
// ---------------------------------------------------------------------------

/// An antichain of failure sets — every member is incomparable (no set is a
/// subset of another).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalFailureSet {
    sets: Vec<FailureSet>,
    capacity: usize,
}

impl MinimalFailureSet {
    pub fn new(capacity: usize) -> Self {
        Self {
            sets: Vec::new(),
            capacity,
        }
    }

    /// Insert a failure set while maintaining the antichain property.
    ///
    /// * If the new set is a superset of any existing member, it is **not** added.
    /// * Any existing member that is a superset of the new set is **removed**.
    pub fn insert(&mut self, set: FailureSet) {
        // If any existing set is a subset of (or equal to) the new one, the new
        // one is redundant — skip it.
        if self.sets.iter().any(|existing| existing.is_subset(&set)) {
            return;
        }
        // Remove any existing set that is a strict superset of the new one.
        self.sets.retain(|existing| !set.is_subset(existing));
        self.sets.push(set);
    }

    pub fn len(&self) -> usize {
        self.sets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &FailureSet> {
        self.sets.iter()
    }

    /// Returns `true` if any member of this antichain is a subset of `set`.
    pub fn contains_subset_of(&self, set: &FailureSet) -> bool {
        self.sets.iter().any(|member| member.is_subset(set))
    }
}

// ---------------------------------------------------------------------------
// CascadeResult
// ---------------------------------------------------------------------------

/// Top-level result of a cascade analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeResult {
    pub scenarios: Vec<CascadeScenario>,
    pub minimal_failure_sets: MinimalFailureSet,
    pub metrics: CascadeMetrics,
    pub analysis_time_ms: u64,
    pub service_index: Vec<ServiceId>,
}

impl CascadeResult {
    /// Returns the scenario with the highest severity (first by ordinal, then
    /// by position for ties).
    pub fn worst_scenario(&self) -> Option<&CascadeScenario> {
        self.scenarios.iter().max_by_key(|s| s.severity)
    }

    pub fn scenario_count(&self) -> usize {
        self.scenarios.len()
    }

    /// `true` if any scenario is classified as `Critical`.
    pub fn has_critical(&self) -> bool {
        self.scenarios
            .iter()
            .any(|s| s.severity == CascadeSeverity::Critical)
    }
}

// ---------------------------------------------------------------------------
// FailureSetOrdering
// ---------------------------------------------------------------------------

/// Utility for comparing failure sets under set-inclusion ordering.
pub struct FailureSetOrdering;

impl FailureSetOrdering {
    pub fn is_subset(a: &FailureSet, b: &FailureSet) -> bool {
        a.is_subset(b)
    }

    pub fn is_superset(a: &FailureSet, b: &FailureSet) -> bool {
        a.is_superset(b)
    }

    /// Partial-order comparison under set inclusion.
    ///
    /// * `Some(Equal)` — identical sets
    /// * `Some(Less)` — `a` is a strict subset of `b`
    /// * `Some(Greater)` — `a` is a strict superset of `b`
    /// * `None` — incomparable
    pub fn compare(a: &FailureSet, b: &FailureSet) -> Option<Ordering> {
        let a_sub_b = a.is_subset(b);
        let b_sub_a = b.is_subset(a);
        match (a_sub_b, b_sub_a) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FailureMode ---------------------------------------------------------

    #[test]
    fn failure_mode_severity_scores() {
        assert_eq!(FailureMode::ServiceDown.severity_score(), 3);
        assert_eq!(
            FailureMode::HighLatency { latency_ms: 500 }.severity_score(),
            2
        );
        assert_eq!(
            FailureMode::PartialFailure { error_rate: 0.1 }.severity_score(),
            1
        );
        assert_eq!(
            FailureMode::ResourceExhaustion { utilization: 0.95 }.severity_score(),
            2
        );
    }

    #[test]
    fn failure_mode_is_total_failure() {
        assert!(FailureMode::ServiceDown.is_total_failure());
        assert!(!FailureMode::HighLatency { latency_ms: 100 }.is_total_failure());
        assert!(!FailureMode::PartialFailure { error_rate: 0.5 }.is_total_failure());
        assert!(!FailureMode::ResourceExhaustion { utilization: 0.9 }.is_total_failure());
    }

    #[test]
    fn failure_mode_display() {
        let mode = FailureMode::HighLatency { latency_ms: 300 };
        assert_eq!(mode.to_string(), "HighLatency(300ms)");
    }

    // -- FailureSet ----------------------------------------------------------

    #[test]
    fn failure_set_insert_remove_contains() {
        let mut fs = FailureSet::new(8);
        assert!(fs.is_empty());
        fs.insert(2);
        fs.insert(5);
        assert!(fs.contains(2));
        assert!(fs.contains(5));
        assert!(!fs.contains(0));
        assert_eq!(fs.len(), 2);
        fs.remove(2);
        assert!(!fs.contains(2));
        assert_eq!(fs.len(), 1);
    }

    #[test]
    fn failure_set_with_failures() {
        let fs = FailureSet::with_failures(10, &[1, 3, 7]);
        assert_eq!(fs.len(), 3);
        assert!(fs.contains(1));
        assert!(fs.contains(3));
        assert!(fs.contains(7));
        assert!(!fs.contains(0));
    }

    #[test]
    fn failure_set_iter_set() {
        let fs = FailureSet::with_failures(8, &[0, 4, 7]);
        let indices: Vec<usize> = fs.iter_set().collect();
        assert_eq!(indices, vec![0, 4, 7]);
    }

    #[test]
    fn failure_set_union() {
        let a = FailureSet::with_failures(8, &[0, 1]);
        let b = FailureSet::with_failures(8, &[1, 2]);
        let u = a.union(&b);
        assert_eq!(u.len(), 3);
        assert!(u.contains(0));
        assert!(u.contains(1));
        assert!(u.contains(2));
    }

    #[test]
    fn failure_set_intersection() {
        let a = FailureSet::with_failures(8, &[0, 1, 2]);
        let b = FailureSet::with_failures(8, &[1, 2, 3]);
        let inter = a.intersection(&b);
        assert_eq!(inter.len(), 2);
        assert!(inter.contains(1));
        assert!(inter.contains(2));
        assert!(!inter.contains(0));
        assert!(!inter.contains(3));
    }

    #[test]
    fn failure_set_subset_superset() {
        let small = FailureSet::with_failures(8, &[1, 3]);
        let big = FailureSet::with_failures(8, &[1, 2, 3, 4]);
        assert!(small.is_subset(&big));
        assert!(!big.is_subset(&small));
        assert!(big.is_superset(&small));
        assert!(!small.is_superset(&big));
    }

    #[test]
    fn failure_set_out_of_bounds_ignored() {
        let mut fs = FailureSet::new(4);
        fs.insert(10); // silently ignored
        assert!(!fs.contains(10));
        assert_eq!(fs.len(), 0);
    }

    // -- PropagationTrace ----------------------------------------------------

    #[test]
    fn propagation_trace_services_affected() {
        let mut trace = PropagationTrace::new();
        trace.push(PropagationStep {
            time_step: 0,
            service: ServiceId::new("svc-a"),
            load: 1.0,
            state: "healthy".into(),
            cause: None,
        });
        trace.push(PropagationStep {
            time_step: 1,
            service: ServiceId::new("svc-b"),
            load: 2.5,
            state: "overloaded".into(),
            cause: Some("svc-a failure".into()),
        });
        trace.push(PropagationStep {
            time_step: 2,
            service: ServiceId::new("svc-a"),
            load: 3.0,
            state: "failed".into(),
            cause: None,
        });
        let affected = trace.services_affected();
        assert_eq!(affected.len(), 2);
        assert!(affected.contains(&ServiceId::new("svc-a")));
        assert!(affected.contains(&ServiceId::new("svc-b")));
    }

    #[test]
    fn propagation_trace_max_load_and_time_to_cascade() {
        let mut trace = PropagationTrace::new();
        trace.push(PropagationStep {
            time_step: 0,
            service: ServiceId::new("a"),
            load: 1.0,
            state: "healthy".into(),
            cause: None,
        });
        trace.push(PropagationStep {
            time_step: 3,
            service: ServiceId::new("b"),
            load: 5.0,
            state: "failed".into(),
            cause: None,
        });
        trace.push(PropagationStep {
            time_step: 5,
            service: ServiceId::new("c"),
            load: 3.0,
            state: "failed".into(),
            cause: None,
        });
        assert!((trace.max_load() - 5.0).abs() < f64::EPSILON);
        assert_eq!(trace.time_to_cascade(), Some(3));
    }

    #[test]
    fn propagation_trace_no_cascade() {
        let mut trace = PropagationTrace::new();
        trace.push(PropagationStep {
            time_step: 0,
            service: ServiceId::new("x"),
            load: 1.0,
            state: "healthy".into(),
            cause: None,
        });
        assert_eq!(trace.time_to_cascade(), None);
    }

    // -- CascadeSeverity -----------------------------------------------------

    #[test]
    fn cascade_severity_ordering() {
        assert!(CascadeSeverity::None < CascadeSeverity::Low);
        assert!(CascadeSeverity::Low < CascadeSeverity::Medium);
        assert!(CascadeSeverity::Medium < CascadeSeverity::High);
        assert!(CascadeSeverity::High < CascadeSeverity::Critical);
    }

    #[test]
    fn cascade_severity_round_trip() {
        for score in 0..=4u8 {
            let sev = CascadeSeverity::from_score(score);
            assert_eq!(sev.score(), score);
        }
        // Saturation for out-of-range
        assert_eq!(CascadeSeverity::from_score(255), CascadeSeverity::Critical);
    }

    #[test]
    fn cascade_severity_display() {
        assert_eq!(CascadeSeverity::Critical.to_string(), "Critical");
        assert_eq!(CascadeSeverity::None.to_string(), "None");
    }

    // -- CascadeClassification -----------------------------------------------

    #[test]
    fn cascade_classification_display_and_description() {
        let c = CascadeClassification::RetryAmplification;
        assert_eq!(c.to_string(), "RetryAmplification");
        assert!(c.description().contains("retry"));

        let combined = CascadeClassification::Combined(vec![
            CascadeClassification::FanInStorm,
            CascadeClassification::TimeoutChainViolation,
        ]);
        let display = combined.to_string();
        assert!(display.contains("FanInStorm"));
        assert!(display.contains("TimeoutChainViolation"));
        assert!(combined.description().contains("combination"));
    }

    // -- CascadeMetrics ------------------------------------------------------

    #[test]
    fn cascade_metrics_severity_thresholds() {
        let none = CascadeMetrics {
            amplification_factor: 1.0,
            max_load: 1.0,
            affected_count: 0,
            time_to_cascade: None,
            total_excess_load: 0.0,
            reliability_impact: 0.0,
        };
        assert_eq!(none.severity(), CascadeSeverity::None);

        let low = CascadeMetrics {
            amplification_factor: 1.5,
            max_load: 1.5,
            affected_count: 1,
            time_to_cascade: Some(10),
            total_excess_load: 0.5,
            reliability_impact: 0.05,
        };
        assert_eq!(low.severity(), CascadeSeverity::Low);

        let medium = CascadeMetrics {
            amplification_factor: 3.0,
            max_load: 3.0,
            affected_count: 3,
            time_to_cascade: Some(5),
            total_excess_load: 2.0,
            reliability_impact: 0.3,
        };
        assert_eq!(medium.severity(), CascadeSeverity::Medium);

        let high = CascadeMetrics {
            amplification_factor: 7.0,
            max_load: 7.0,
            affected_count: 5,
            time_to_cascade: Some(2),
            total_excess_load: 6.0,
            reliability_impact: 0.6,
        };
        assert_eq!(high.severity(), CascadeSeverity::High);

        let critical = CascadeMetrics {
            amplification_factor: 12.0,
            max_load: 12.0,
            affected_count: 10,
            time_to_cascade: Some(1),
            total_excess_load: 11.0,
            reliability_impact: 0.9,
        };
        assert_eq!(critical.severity(), CascadeSeverity::Critical);
    }

    // -- MinimalFailureSet ---------------------------------------------------

    #[test]
    fn minimal_failure_set_antichain_property() {
        let mut mfs = MinimalFailureSet::new(8);

        // Insert {0, 1}
        mfs.insert(FailureSet::with_failures(8, &[0, 1]));
        assert_eq!(mfs.len(), 1);

        // Insert {0, 1, 2} — superset of {0,1} so should be rejected
        mfs.insert(FailureSet::with_failures(8, &[0, 1, 2]));
        assert_eq!(mfs.len(), 1);

        // Insert {3} — incomparable, should be added
        mfs.insert(FailureSet::with_failures(8, &[3]));
        assert_eq!(mfs.len(), 2);

        // Insert {0} — subset of {0,1}, so {0,1} should be removed
        mfs.insert(FailureSet::with_failures(8, &[0]));
        assert_eq!(mfs.len(), 2); // {0} and {3}
        assert!(mfs.contains_subset_of(&FailureSet::with_failures(8, &[0, 1, 2])));
    }

    #[test]
    fn minimal_failure_set_contains_subset_of() {
        let mut mfs = MinimalFailureSet::new(8);
        mfs.insert(FailureSet::with_failures(8, &[1, 2]));

        assert!(mfs.contains_subset_of(&FailureSet::with_failures(8, &[1, 2, 3])));
        assert!(!mfs.contains_subset_of(&FailureSet::with_failures(8, &[0, 3])));
    }

    // -- CascadeResult -------------------------------------------------------

    #[test]
    fn cascade_result_worst_scenario() {
        let s1 = CascadeScenario::builder()
            .id("s1")
            .severity(CascadeSeverity::Low)
            .description("low scenario")
            .build();
        let s2 = CascadeScenario::builder()
            .id("s2")
            .severity(CascadeSeverity::Critical)
            .description("critical scenario")
            .build();
        let s3 = CascadeScenario::builder()
            .id("s3")
            .severity(CascadeSeverity::Medium)
            .description("medium scenario")
            .build();

        let result = CascadeResult {
            scenarios: vec![s1, s2, s3],
            minimal_failure_sets: MinimalFailureSet::new(8),
            metrics: CascadeMetrics {
                amplification_factor: 1.0,
                max_load: 1.0,
                affected_count: 0,
                time_to_cascade: None,
                total_excess_load: 0.0,
                reliability_impact: 0.0,
            },
            analysis_time_ms: 42,
            service_index: vec![],
        };

        assert_eq!(result.scenario_count(), 3);
        assert!(result.has_critical());
        let worst = result.worst_scenario().unwrap();
        assert_eq!(worst.id, "s2");
    }

    // -- FailureSetOrdering --------------------------------------------------

    #[test]
    fn failure_set_ordering_compare() {
        let a = FailureSet::with_failures(8, &[1]);
        let b = FailureSet::with_failures(8, &[1, 2]);
        let c = FailureSet::with_failures(8, &[3]);
        let d = FailureSet::with_failures(8, &[1]);

        assert_eq!(FailureSetOrdering::compare(&a, &b), Some(Ordering::Less));
        assert_eq!(FailureSetOrdering::compare(&b, &a), Some(Ordering::Greater));
        assert_eq!(FailureSetOrdering::compare(&a, &c), Option::None);
        assert_eq!(FailureSetOrdering::compare(&a, &d), Some(Ordering::Equal));

        assert!(FailureSetOrdering::is_subset(&a, &b));
        assert!(FailureSetOrdering::is_superset(&b, &a));
    }

    // -- Serialization -------------------------------------------------------

    #[test]
    fn failure_set_serde_round_trip() {
        let fs = FailureSet::with_failures(16, &[2, 5, 11]);
        let json = serde_json::to_string(&fs).unwrap();
        let deserialized: FailureSet = serde_json::from_str(&json).unwrap();
        assert_eq!(fs, deserialized);
    }

    #[test]
    fn failure_mode_serde_round_trip() {
        let mode = FailureMode::HighLatency { latency_ms: 250 };
        let json = serde_json::to_string(&mode).unwrap();
        let deserialized: FailureMode = serde_json::from_str(&json).unwrap();
        assert_eq!(mode, deserialized);
    }

    #[test]
    fn cascade_severity_serde_round_trip() {
        let sev = CascadeSeverity::High;
        let json = serde_json::to_string(&sev).unwrap();
        let deserialized: CascadeSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(sev, deserialized);
    }

    #[test]
    fn propagation_step_display() {
        let step = PropagationStep {
            time_step: 7,
            service: ServiceId::new("gateway"),
            load: 4.25,
            state: "overloaded".into(),
            cause: Some("backend timeout".into()),
        };
        let display = step.to_string();
        assert!(display.contains("t=7"));
        assert!(display.contains("gateway"));
        assert!(display.contains("4.25"));
        assert!(display.contains("backend timeout"));
    }
}

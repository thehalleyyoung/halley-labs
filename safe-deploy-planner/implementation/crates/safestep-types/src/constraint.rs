// Constraint types for the SafeStep deployment planner.

use std::fmt;

use bitvec::prelude::*;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::graph::ClusterState;
use crate::identifiers::ConstraintId;
use crate::service::ResourceQuantity;
use crate::version::{VersionIndex, VersionRange};

// ─── Constraint strength ────────────────────────────────────────────────

/// How strictly a constraint must be enforced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintStrength {
    /// Must be satisfied; violation makes the plan infeasible.
    Hard,
    /// Soft constraint with a penalty weight.
    Soft(OrderedFloat<f64>),
    /// Preference with priority level (higher = more important).
    Preference(u32),
}

impl ConstraintStrength {
    pub fn is_hard(&self) -> bool {
        matches!(self, Self::Hard)
    }

    pub fn weight(&self) -> f64 {
        match self {
            Self::Hard => f64::INFINITY,
            Self::Soft(w) => w.into_inner(),
            Self::Preference(p) => *p as f64,
        }
    }
}

impl fmt::Display for ConstraintStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hard => write!(f, "hard"),
            Self::Soft(w) => write!(f, "soft({})", w),
            Self::Preference(p) => write!(f, "pref({})", p),
        }
    }
}

// ─── Constraint status ──────────────────────────────────────────────────

/// Result of evaluating a constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintStatus {
    Satisfied,
    Violated,
    Unknown,
}

impl ConstraintStatus {
    pub fn is_satisfied(&self) -> bool {
        *self == Self::Satisfied
    }

    pub fn is_violated(&self) -> bool {
        *self == Self::Violated
    }

    /// Combine two statuses: Violated trumps Unknown trumps Satisfied.
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::Violated, _) | (_, Self::Violated) => Self::Violated,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            _ => Self::Satisfied,
        }
    }
}

impl fmt::Display for ConstraintStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Satisfied => write!(f, "satisfied"),
            Self::Violated => write!(f, "VIOLATED"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ─── Constraint evaluation ──────────────────────────────────────────────

/// Full result of evaluating a constraint on a state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintEvaluation {
    pub constraint_id: ConstraintId,
    pub status: ConstraintStatus,
    pub strength: ConstraintStrength,
    pub violation_detail: Option<String>,
    pub penalty: OrderedFloat<f64>,
}

impl ConstraintEvaluation {
    pub fn satisfied(id: ConstraintId, strength: ConstraintStrength) -> Self {
        Self {
            constraint_id: id,
            status: ConstraintStatus::Satisfied,
            strength,
            violation_detail: None,
            penalty: OrderedFloat(0.0),
        }
    }

    pub fn violated(
        id: ConstraintId,
        strength: ConstraintStrength,
        detail: impl Into<String>,
    ) -> Self {
        let penalty = OrderedFloat(strength.weight());
        Self {
            constraint_id: id,
            status: ConstraintStatus::Violated,
            strength,
            violation_detail: Some(detail.into()),
            penalty,
        }
    }

    pub fn unknown(id: ConstraintId, strength: ConstraintStrength) -> Self {
        Self {
            constraint_id: id,
            status: ConstraintStatus::Unknown,
            strength,
            violation_detail: None,
            penalty: OrderedFloat(0.0),
        }
    }
}

impl fmt::Display for ConstraintEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} [{}]",
            self.constraint_id, self.status, self.strength
        )?;
        if let Some(detail) = &self.violation_detail {
            write!(f, " - {}", detail)?;
        }
        Ok(())
    }
}

// ─── Compatibility constraint ────────────────────────────────────────────

/// Pairwise compatibility between two services at specific versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityConstraint {
    pub id: ConstraintId,
    pub service_a: usize,
    pub service_b: usize,
    pub compatible_pairs: Vec<(VersionIndex, VersionIndex)>,
    pub strength: ConstraintStrength,
    pub description: String,
}

impl CompatibilityConstraint {
    pub fn new(
        id: ConstraintId,
        service_a: usize,
        service_b: usize,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id,
            service_a,
            service_b,
            compatible_pairs: Vec::new(),
            strength: ConstraintStrength::Hard,
            description: description.into(),
        }
    }

    pub fn add_compatible_pair(&mut self, va: VersionIndex, vb: VersionIndex) {
        self.compatible_pairs.push((va, vb));
    }

    pub fn with_strength(mut self, strength: ConstraintStrength) -> Self {
        self.strength = strength;
        self
    }

    pub fn is_compatible(&self, va: VersionIndex, vb: VersionIndex) -> bool {
        self.compatible_pairs.iter().any(|&(a, b)| a == va && b == vb)
    }

    /// Evaluate on a cluster state.
    pub fn evaluate(&self, state: &ClusterState) -> ConstraintEvaluation {
        let va = state.get(self.service_a);
        let vb = state.get(self.service_b);
        if self.is_compatible(va, vb) {
            ConstraintEvaluation::satisfied(self.id.clone(), self.strength.clone())
        } else {
            ConstraintEvaluation::violated(
                self.id.clone(),
                self.strength.clone(),
                format!(
                    "Services {} and {} at versions ({}, {}) are not compatible",
                    self.service_a, self.service_b, va, vb
                ),
            )
        }
    }

    /// Convert to a CompatibilityZone (bitmap encoding).
    pub fn to_zone(
        &self,
        max_a: u32,
        max_b: u32,
    ) -> CompatibilityZone {
        let width = max_a as usize;
        let height = max_b as usize;
        let mut bits = bitvec![u64, Lsb0; 0; width * height];
        for &(va, vb) in &self.compatible_pairs {
            let idx = va.0 as usize * height + vb.0 as usize;
            if idx < bits.len() {
                bits.set(idx, true);
            }
        }
        CompatibilityZone {
            service_a: self.service_a,
            service_b: self.service_b,
            width,
            height,
            bits,
        }
    }
}

impl fmt::Display for CompatibilityConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Compat({}->{}, {} pairs) [{}]: {}",
            self.service_a,
            self.service_b,
            self.compatible_pairs.len(),
            self.strength,
            self.description,
        )
    }
}

// ─── Interval constraint ────────────────────────────────────────────────

/// A compatibility constraint with interval structure: for each version of
/// service A, there is a contiguous range [lo, hi] of compatible versions
/// of service B. This enables efficient SAT encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalConstraint {
    pub id: ConstraintId,
    pub service_a: usize,
    pub service_b: usize,
    /// For each version index of service A, the compatible range of B.
    pub intervals: Vec<Option<VersionRange>>,
    pub strength: ConstraintStrength,
    pub description: String,
}

impl IntervalConstraint {
    pub fn new(
        id: ConstraintId,
        service_a: usize,
        service_b: usize,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id,
            service_a,
            service_b,
            intervals: Vec::new(),
            strength: ConstraintStrength::Hard,
            description: description.into(),
        }
    }

    pub fn set_interval(&mut self, va: VersionIndex, range: VersionRange) {
        let idx = va.as_usize();
        if idx >= self.intervals.len() {
            self.intervals.resize(idx + 1, None);
        }
        self.intervals[idx] = Some(range);
    }

    pub fn with_strength(mut self, strength: ConstraintStrength) -> Self {
        self.strength = strength;
        self
    }

    pub fn is_compatible(&self, va: VersionIndex, vb: VersionIndex) -> bool {
        self.intervals
            .get(va.as_usize())
            .and_then(|r| r.as_ref())
            .map(|range| range.contains(vb))
            .unwrap_or(false)
    }

    pub fn evaluate(&self, state: &ClusterState) -> ConstraintEvaluation {
        let va = state.get(self.service_a);
        let vb = state.get(self.service_b);
        if self.is_compatible(va, vb) {
            ConstraintEvaluation::satisfied(self.id.clone(), self.strength.clone())
        } else {
            ConstraintEvaluation::violated(
                self.id.clone(),
                self.strength.clone(),
                format!(
                    "Version {} of service {} is outside interval for version {} of service {}",
                    vb, self.service_b, va, self.service_a
                ),
            )
        }
    }

    /// Check if this constraint has interval structure (all ranges are contiguous).
    pub fn has_interval_structure(&self) -> bool {
        self.intervals.iter().all(|r| r.is_some() || true)
    }

    /// Number of compatible pairs implied by the intervals.
    pub fn compatible_pair_count(&self) -> u64 {
        self.intervals
            .iter()
            .filter_map(|r| r.as_ref())
            .map(|r| r.len() as u64)
            .sum()
    }

    /// Convert to a plain CompatibilityConstraint.
    pub fn to_compatibility(&self) -> CompatibilityConstraint {
        let mut cc = CompatibilityConstraint::new(
            self.id.clone(),
            self.service_a,
            self.service_b,
            &self.description,
        );
        cc.strength = self.strength.clone();
        for (a_idx, interval) in self.intervals.iter().enumerate() {
            if let Some(range) = interval {
                for b_idx in range.iter() {
                    cc.add_compatible_pair(VersionIndex(a_idx as u32), b_idx);
                }
            }
        }
        cc
    }
}

impl fmt::Display for IntervalConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Interval({}->{}, {} intervals) [{}]: {}",
            self.service_a,
            self.service_b,
            self.intervals.iter().filter(|i| i.is_some()).count(),
            self.strength,
            self.description,
        )
    }
}

// ─── Resource constraint ────────────────────────────────────────────────

/// Linear arithmetic constraint over resource quantities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraint {
    pub id: ConstraintId,
    pub resource_name: String,
    /// Per-service per-version resource usage: coefficients[service_idx][version_idx]
    pub coefficients: Vec<Vec<ResourceQuantity>>,
    pub bound: ResourceQuantity,
    pub comparison: ComparisonOp,
    pub strength: ConstraintStrength,
    pub description: String,
}

/// Comparison operator for resource constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComparisonOp {
    LessEqual,
    GreaterEqual,
    Equal,
    Less,
    Greater,
}

impl ComparisonOp {
    pub fn evaluate(&self, lhs: u64, rhs: u64) -> bool {
        match self {
            Self::LessEqual => lhs <= rhs,
            Self::GreaterEqual => lhs >= rhs,
            Self::Equal => lhs == rhs,
            Self::Less => lhs < rhs,
            Self::Greater => lhs > rhs,
        }
    }
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LessEqual => write!(f, "<="),
            Self::GreaterEqual => write!(f, ">="),
            Self::Equal => write!(f, "=="),
            Self::Less => write!(f, "<"),
            Self::Greater => write!(f, ">"),
        }
    }
}

impl ResourceConstraint {
    pub fn new(
        id: ConstraintId,
        resource_name: impl Into<String>,
        bound: ResourceQuantity,
        comparison: ComparisonOp,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id,
            resource_name: resource_name.into(),
            coefficients: Vec::new(),
            bound,
            comparison,
            strength: ConstraintStrength::Hard,
            description: description.into(),
        }
    }

    pub fn set_coefficients(&mut self, service_idx: usize, coeffs: Vec<ResourceQuantity>) {
        if service_idx >= self.coefficients.len() {
            self.coefficients
                .resize(service_idx + 1, Vec::new());
        }
        self.coefficients[service_idx] = coeffs;
    }

    pub fn with_strength(mut self, strength: ConstraintStrength) -> Self {
        self.strength = strength;
        self
    }

    pub fn evaluate(&self, state: &ClusterState) -> ConstraintEvaluation {
        let mut total = 0u64;
        for (svc_idx, coeffs) in self.coefficients.iter().enumerate() {
            if svc_idx < state.num_services() {
                let vi = state.get(svc_idx).as_usize();
                if vi < coeffs.len() {
                    total = total.saturating_add(coeffs[vi].millicores);
                }
            }
        }
        if self.comparison.evaluate(total, self.bound.millicores) {
            ConstraintEvaluation::satisfied(self.id.clone(), self.strength.clone())
        } else {
            ConstraintEvaluation::violated(
                self.id.clone(),
                self.strength.clone(),
                format!(
                    "Resource '{}': {} {} {} = false",
                    self.resource_name, total, self.comparison, self.bound.millicores
                ),
            )
        }
    }
}

impl fmt::Display for ResourceConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Resource({} {} {}) [{}]: {}",
            self.resource_name, self.comparison, self.bound, self.strength, self.description
        )
    }
}

// ─── Constraint enum ────────────────────────────────────────────────────

/// Top-level constraint type, unifying all constraint kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    Compatibility(CompatibilityConstraint),
    Interval(IntervalConstraint),
    Resource(ResourceConstraint),
    Custom(CustomConstraint),
}

impl Constraint {
    pub fn id(&self) -> &ConstraintId {
        match self {
            Self::Compatibility(c) => &c.id,
            Self::Interval(c) => &c.id,
            Self::Resource(c) => &c.id,
            Self::Custom(c) => &c.id,
        }
    }

    pub fn strength(&self) -> &ConstraintStrength {
        match self {
            Self::Compatibility(c) => &c.strength,
            Self::Interval(c) => &c.strength,
            Self::Resource(c) => &c.strength,
            Self::Custom(c) => &c.strength,
        }
    }

    pub fn evaluate(&self, state: &ClusterState) -> ConstraintEvaluation {
        match self {
            Self::Compatibility(c) => c.evaluate(state),
            Self::Interval(c) => c.evaluate(state),
            Self::Resource(c) => c.evaluate(state),
            Self::Custom(c) => c.evaluate(state),
        }
    }

    pub fn description(&self) -> &str {
        match self {
            Self::Compatibility(c) => &c.description,
            Self::Interval(c) => &c.description,
            Self::Resource(c) => &c.description,
            Self::Custom(c) => &c.description,
        }
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compatibility(c) => write!(f, "{}", c),
            Self::Interval(c) => write!(f, "{}", c),
            Self::Resource(c) => write!(f, "{}", c),
            Self::Custom(c) => write!(f, "{}", c),
        }
    }
}

// ─── Custom constraint ──────────────────────────────────────────────────

/// A custom constraint defined by a set of allowed states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomConstraint {
    pub id: ConstraintId,
    pub description: String,
    pub strength: ConstraintStrength,
    /// Set of version-index tuples: each inner Vec corresponds to services;
    /// the state must match one of these tuples.
    pub allowed_states: Vec<Vec<VersionIndex>>,
}

impl CustomConstraint {
    pub fn new(id: ConstraintId, description: impl Into<String>) -> Self {
        Self {
            id,
            description: description.into(),
            strength: ConstraintStrength::Hard,
            allowed_states: Vec::new(),
        }
    }

    pub fn add_allowed(&mut self, state: Vec<VersionIndex>) {
        self.allowed_states.push(state);
    }

    pub fn with_strength(mut self, strength: ConstraintStrength) -> Self {
        self.strength = strength;
        self
    }

    pub fn evaluate(&self, state: &ClusterState) -> ConstraintEvaluation {
        let current: Vec<VersionIndex> = (0..state.num_services())
            .map(|i| state.get(i))
            .collect();
        if self.allowed_states.is_empty() || self.allowed_states.iter().any(|a| *a == current) {
            ConstraintEvaluation::satisfied(self.id.clone(), self.strength.clone())
        } else {
            ConstraintEvaluation::violated(
                self.id.clone(),
                self.strength.clone(),
                format!("State {:?} not in allowed set", current),
            )
        }
    }
}

impl fmt::Display for CustomConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Custom({}, {} allowed) [{}]: {}",
            self.id,
            self.allowed_states.len(),
            self.strength,
            self.description,
        )
    }
}

// ─── ConstraintSet ──────────────────────────────────────────────────────

/// A collection of constraints with indexing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintSet {
    constraints: Vec<Constraint>,
    index_by_id: indexmap::IndexMap<String, usize>,
}

impl ConstraintSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, constraint: Constraint) {
        let id = constraint.id().as_str().to_string();
        let idx = self.constraints.len();
        self.constraints.push(constraint);
        self.index_by_id.insert(id, idx);
    }

    pub fn get(&self, id: &str) -> Option<&Constraint> {
        self.index_by_id
            .get(id)
            .and_then(|&idx| self.constraints.get(idx))
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.iter()
    }

    pub fn hard_constraints(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints
            .iter()
            .filter(|c| c.strength().is_hard())
    }

    pub fn soft_constraints(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints
            .iter()
            .filter(|c| !c.strength().is_hard())
    }

    /// Evaluate all constraints on a state.
    pub fn evaluate_all(&self, state: &ClusterState) -> Vec<ConstraintEvaluation> {
        self.constraints.iter().map(|c| c.evaluate(state)).collect()
    }

    /// Check if all hard constraints are satisfied.
    pub fn is_safe(&self, state: &ClusterState) -> bool {
        self.hard_constraints()
            .all(|c| c.evaluate(state).status.is_satisfied())
    }

    /// Return all violated hard constraints.
    pub fn violations(&self, state: &ClusterState) -> Vec<ConstraintEvaluation> {
        self.evaluate_all(state)
            .into_iter()
            .filter(|e| e.status.is_violated() && e.strength.is_hard())
            .collect()
    }

    /// Total penalty from soft constraint violations.
    pub fn total_penalty(&self, state: &ClusterState) -> f64 {
        self.evaluate_all(state)
            .iter()
            .map(|e| e.penalty.into_inner())
            .sum()
    }

    /// Constraint IDs for constraints involving a specific service.
    pub fn constraints_for_service(&self, service_idx: usize) -> Vec<&Constraint> {
        self.constraints
            .iter()
            .filter(|c| match c {
                Constraint::Compatibility(cc) => {
                    cc.service_a == service_idx || cc.service_b == service_idx
                }
                Constraint::Interval(ic) => {
                    ic.service_a == service_idx || ic.service_b == service_idx
                }
                Constraint::Resource(rc) => service_idx < rc.coefficients.len(),
                Constraint::Custom(_) => true,
            })
            .collect()
    }

    /// Merge another constraint set into this one.
    pub fn merge(&mut self, other: ConstraintSet) {
        for c in other.constraints {
            self.add(c);
        }
    }
}

// ─── CompatibilityZone ──────────────────────────────────────────────────

/// Bitmap representation of compatible version pairs for two services.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityZone {
    pub service_a: usize,
    pub service_b: usize,
    pub width: usize,
    pub height: usize,
    bits: BitVec<u64, Lsb0>,
}

impl CompatibilityZone {
    pub fn new(service_a: usize, service_b: usize, width: usize, height: usize) -> Self {
        Self {
            service_a,
            service_b,
            width,
            height,
            bits: bitvec![u64, Lsb0; 0; width * height],
        }
    }

    pub fn set(&mut self, va: usize, vb: usize, compatible: bool) {
        let idx = va * self.height + vb;
        if idx < self.bits.len() {
            self.bits.set(idx, compatible);
        }
    }

    pub fn get(&self, va: usize, vb: usize) -> bool {
        let idx = va * self.height + vb;
        self.bits.get(idx).map(|b| *b).unwrap_or(false)
    }

    pub fn is_compatible(&self, va: VersionIndex, vb: VersionIndex) -> bool {
        self.get(va.as_usize(), vb.as_usize())
    }

    /// Count of compatible pairs.
    pub fn compatible_count(&self) -> usize {
        self.bits.count_ones()
    }

    /// Total pairs.
    pub fn total_pairs(&self) -> usize {
        self.width * self.height
    }

    /// Compatibility ratio.
    pub fn compatibility_ratio(&self) -> f64 {
        let total = self.total_pairs();
        if total == 0 {
            0.0
        } else {
            self.compatible_count() as f64 / total as f64
        }
    }

    /// Check if the zone has interval structure (each row is a contiguous range).
    pub fn has_interval_structure(&self) -> bool {
        for va in 0..self.width {
            let mut found_start = false;
            let mut found_end = false;
            for vb in 0..self.height {
                let compat = self.get(va, vb);
                if compat && found_end {
                    return false; // non-contiguous
                }
                if compat && !found_start {
                    found_start = true;
                }
                if !compat && found_start && !found_end {
                    found_end = true;
                }
            }
        }
        true
    }

    /// Convert to interval constraint if it has interval structure.
    pub fn to_interval_constraint(
        &self,
        id: ConstraintId,
        description: impl Into<String>,
    ) -> Option<IntervalConstraint> {
        if !self.has_interval_structure() {
            return None;
        }
        let mut ic = IntervalConstraint::new(id, self.service_a, self.service_b, description);
        for va in 0..self.width {
            let mut lo = None;
            let mut hi = None;
            for vb in 0..self.height {
                if self.get(va, vb) {
                    if lo.is_none() {
                        lo = Some(vb);
                    }
                    hi = Some(vb);
                }
            }
            if let (Some(l), Some(h)) = (lo, hi) {
                ic.set_interval(
                    VersionIndex(va as u32),
                    VersionRange::new(VersionIndex(l as u32), VersionIndex(h as u32)),
                );
            }
        }
        Some(ic)
    }

    /// AND of two zones (both must be compatible).
    pub fn intersection(&self, other: &CompatibilityZone) -> CompatibilityZone {
        debug_assert_eq!(self.width, other.width);
        debug_assert_eq!(self.height, other.height);
        let mut result = CompatibilityZone::new(
            self.service_a,
            self.service_b,
            self.width,
            self.height,
        );
        result.bits = self.bits.clone() & other.bits.clone();
        result
    }

    /// OR of two zones (either must be compatible).
    pub fn union(&self, other: &CompatibilityZone) -> CompatibilityZone {
        debug_assert_eq!(self.width, other.width);
        debug_assert_eq!(self.height, other.height);
        let mut result = CompatibilityZone::new(
            self.service_a,
            self.service_b,
            self.width,
            self.height,
        );
        result.bits = self.bits.clone() | other.bits.clone();
        result
    }
}

impl fmt::Display for CompatibilityZone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Zone({}x{}, {}/{} compatible, {:.1}%)",
            self.width,
            self.height,
            self.compatible_count(),
            self.total_pairs(),
            self.compatibility_ratio() * 100.0,
        )
    }
}

// ─── PairwiseCompatibility ──────────────────────────────────────────────

/// Full pairwise compatibility matrix between all pairs of services.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseCompatibility {
    pub num_services: usize,
    pub zones: Vec<CompatibilityZone>,
}

impl PairwiseCompatibility {
    pub fn new(num_services: usize) -> Self {
        Self {
            num_services,
            zones: Vec::new(),
        }
    }

    pub fn add_zone(&mut self, zone: CompatibilityZone) {
        self.zones.push(zone);
    }

    pub fn get_zone(&self, service_a: usize, service_b: usize) -> Option<&CompatibilityZone> {
        self.zones
            .iter()
            .find(|z| z.service_a == service_a && z.service_b == service_b)
    }

    pub fn is_fully_compatible(&self, state: &ClusterState) -> bool {
        self.zones.iter().all(|z| {
            let va = state.get(z.service_a);
            let vb = state.get(z.service_b);
            z.is_compatible(va, vb)
        })
    }

    /// Count of constraint pairs that have interval structure.
    pub fn interval_structure_count(&self) -> usize {
        self.zones.iter().filter(|z| z.has_interval_structure()).count()
    }
}

/// Try to infer interval structure from a compatibility constraint.
pub fn infer_interval_structure(
    constraint: &CompatibilityConstraint,
    max_a: u32,
    max_b: u32,
) -> Option<IntervalConstraint> {
    let zone = constraint.to_zone(max_a, max_b);
    zone.to_interval_constraint(constraint.id.clone(), &constraint.description)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_strength() {
        assert!(ConstraintStrength::Hard.is_hard());
        assert!(!ConstraintStrength::Soft(OrderedFloat(1.0)).is_hard());
        assert_eq!(ConstraintStrength::Soft(OrderedFloat(2.5)).weight(), 2.5);
    }

    #[test]
    fn test_constraint_status_combine() {
        assert_eq!(
            ConstraintStatus::Satisfied.combine(ConstraintStatus::Satisfied),
            ConstraintStatus::Satisfied
        );
        assert_eq!(
            ConstraintStatus::Satisfied.combine(ConstraintStatus::Violated),
            ConstraintStatus::Violated
        );
        assert_eq!(
            ConstraintStatus::Unknown.combine(ConstraintStatus::Satisfied),
            ConstraintStatus::Unknown
        );
    }

    #[test]
    fn test_compatibility_constraint() {
        let mut cc = CompatibilityConstraint::new(
            ConstraintId::new("c1"),
            0,
            1,
            "API compat",
        );
        cc.add_compatible_pair(VersionIndex(0), VersionIndex(0));
        cc.add_compatible_pair(VersionIndex(0), VersionIndex(1));
        cc.add_compatible_pair(VersionIndex(1), VersionIndex(1));

        assert!(cc.is_compatible(VersionIndex(0), VersionIndex(0)));
        assert!(cc.is_compatible(VersionIndex(0), VersionIndex(1)));
        assert!(cc.is_compatible(VersionIndex(1), VersionIndex(1)));
        assert!(!cc.is_compatible(VersionIndex(1), VersionIndex(0)));
    }

    #[test]
    fn test_compatibility_evaluate() {
        let mut cc = CompatibilityConstraint::new(
            ConstraintId::new("c1"),
            0,
            1,
            "test",
        );
        cc.add_compatible_pair(VersionIndex(0), VersionIndex(0));

        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        let eval = cc.evaluate(&s1);
        assert!(eval.status.is_satisfied());

        let s2 = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        let eval = cc.evaluate(&s2);
        assert!(eval.status.is_violated());
    }

    #[test]
    fn test_interval_constraint() {
        let mut ic = IntervalConstraint::new(
            ConstraintId::new("ic1"),
            0,
            1,
            "interval compat",
        );
        ic.set_interval(
            VersionIndex(0),
            VersionRange::new(VersionIndex(0), VersionIndex(2)),
        );
        ic.set_interval(
            VersionIndex(1),
            VersionRange::new(VersionIndex(1), VersionIndex(2)),
        );

        assert!(ic.is_compatible(VersionIndex(0), VersionIndex(0)));
        assert!(ic.is_compatible(VersionIndex(0), VersionIndex(1)));
        assert!(ic.is_compatible(VersionIndex(0), VersionIndex(2)));
        assert!(!ic.is_compatible(VersionIndex(1), VersionIndex(0)));
        assert!(ic.is_compatible(VersionIndex(1), VersionIndex(1)));
    }

    #[test]
    fn test_interval_to_compatibility() {
        let mut ic = IntervalConstraint::new(
            ConstraintId::new("ic"),
            0,
            1,
            "test",
        );
        ic.set_interval(
            VersionIndex(0),
            VersionRange::new(VersionIndex(0), VersionIndex(1)),
        );
        let cc = ic.to_compatibility();
        assert_eq!(cc.compatible_pairs.len(), 2);
    }

    #[test]
    fn test_resource_constraint() {
        let mut rc = ResourceConstraint::new(
            ConstraintId::new("r1"),
            "cpu",
            ResourceQuantity::from_millicores(1000),
            ComparisonOp::LessEqual,
            "total CPU <= 1000m",
        );
        rc.set_coefficients(
            0,
            vec![
                ResourceQuantity::from_millicores(200),
                ResourceQuantity::from_millicores(400),
            ],
        );
        rc.set_coefficients(
            1,
            vec![
                ResourceQuantity::from_millicores(300),
                ResourceQuantity::from_millicores(500),
            ],
        );

        // state (0, 0): 200 + 300 = 500 <= 1000
        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        assert!(rc.evaluate(&s1).status.is_satisfied());

        // state (1, 1): 400 + 500 = 900 <= 1000
        let s2 = ClusterState::new(&[VersionIndex(1), VersionIndex(1)]);
        assert!(rc.evaluate(&s2).status.is_satisfied());
    }

    #[test]
    fn test_resource_constraint_violated() {
        let mut rc = ResourceConstraint::new(
            ConstraintId::new("r1"),
            "cpu",
            ResourceQuantity::from_millicores(500),
            ComparisonOp::LessEqual,
            "tight bound",
        );
        rc.set_coefficients(
            0,
            vec![ResourceQuantity::from_millicores(300)],
        );
        rc.set_coefficients(
            1,
            vec![ResourceQuantity::from_millicores(300)],
        );

        let s = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        let eval = rc.evaluate(&s);
        assert!(eval.status.is_violated());
    }

    #[test]
    fn test_comparison_op() {
        assert!(ComparisonOp::LessEqual.evaluate(5, 5));
        assert!(ComparisonOp::LessEqual.evaluate(4, 5));
        assert!(!ComparisonOp::LessEqual.evaluate(6, 5));
        assert!(ComparisonOp::Equal.evaluate(5, 5));
        assert!(!ComparisonOp::Equal.evaluate(4, 5));
    }

    #[test]
    fn test_constraint_set() {
        let mut cs = ConstraintSet::new();
        let mut cc = CompatibilityConstraint::new(
            ConstraintId::new("c1"),
            0,
            1,
            "test",
        );
        cc.add_compatible_pair(VersionIndex(0), VersionIndex(0));
        cs.add(Constraint::Compatibility(cc));

        assert_eq!(cs.len(), 1);
        assert!(cs.get("c1").is_some());

        let state = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        assert!(cs.is_safe(&state));

        let state2 = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        assert!(!cs.is_safe(&state2));
    }

    #[test]
    fn test_constraint_set_violations() {
        let mut cs = ConstraintSet::new();
        let mut cc = CompatibilityConstraint::new(
            ConstraintId::new("c1"),
            0,
            1,
            "test",
        );
        cc.add_compatible_pair(VersionIndex(0), VersionIndex(0));
        cs.add(Constraint::Compatibility(cc));

        let state = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        let violations = cs.violations(&state);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_compatibility_zone() {
        let mut zone = CompatibilityZone::new(0, 1, 3, 3);
        zone.set(0, 0, true);
        zone.set(0, 1, true);
        zone.set(1, 1, true);
        zone.set(1, 2, true);
        zone.set(2, 2, true);

        assert!(zone.get(0, 0));
        assert!(zone.get(0, 1));
        assert!(!zone.get(0, 2));
        assert_eq!(zone.compatible_count(), 5);
        assert!(zone.has_interval_structure());
    }

    #[test]
    fn test_zone_to_interval() {
        let mut zone = CompatibilityZone::new(0, 1, 2, 3);
        zone.set(0, 0, true);
        zone.set(0, 1, true);
        zone.set(0, 2, true);
        zone.set(1, 1, true);
        zone.set(1, 2, true);

        let ic = zone.to_interval_constraint(
            ConstraintId::new("test"),
            "test interval",
        );
        assert!(ic.is_some());
        let ic = ic.unwrap();
        assert!(ic.is_compatible(VersionIndex(0), VersionIndex(0)));
        assert!(ic.is_compatible(VersionIndex(0), VersionIndex(2)));
        assert!(ic.is_compatible(VersionIndex(1), VersionIndex(1)));
        assert!(!ic.is_compatible(VersionIndex(1), VersionIndex(0)));
    }

    #[test]
    fn test_zone_no_interval() {
        let mut zone = CompatibilityZone::new(0, 1, 2, 3);
        zone.set(0, 0, true);
        zone.set(0, 2, true); // gap
        // Row 0: compatible at 0 and 2 but not 1 — not interval
        assert!(!zone.has_interval_structure());
        assert!(zone.to_interval_constraint(ConstraintId::new("x"), "").is_none());
    }

    #[test]
    fn test_zone_intersection() {
        let mut z1 = CompatibilityZone::new(0, 1, 2, 2);
        z1.set(0, 0, true);
        z1.set(0, 1, true);
        z1.set(1, 0, true);

        let mut z2 = CompatibilityZone::new(0, 1, 2, 2);
        z2.set(0, 1, true);
        z2.set(1, 0, true);
        z2.set(1, 1, true);

        let inter = z1.intersection(&z2);
        assert!(inter.get(0, 1));
        assert!(inter.get(1, 0));
        assert!(!inter.get(0, 0));
        assert!(!inter.get(1, 1));
    }

    #[test]
    fn test_zone_union() {
        let mut z1 = CompatibilityZone::new(0, 1, 2, 2);
        z1.set(0, 0, true);

        let mut z2 = CompatibilityZone::new(0, 1, 2, 2);
        z2.set(1, 1, true);

        let union = z1.union(&z2);
        assert!(union.get(0, 0));
        assert!(union.get(1, 1));
        assert_eq!(union.compatible_count(), 2);
    }

    #[test]
    fn test_pairwise_compatibility() {
        let mut pw = PairwiseCompatibility::new(3);
        let mut z = CompatibilityZone::new(0, 1, 2, 2);
        z.set(0, 0, true);
        z.set(1, 1, true);
        pw.add_zone(z);

        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(0), VersionIndex(0)]);
        assert!(pw.is_fully_compatible(&s1));

        let s2 = ClusterState::new(&[VersionIndex(0), VersionIndex(1), VersionIndex(0)]);
        assert!(!pw.is_fully_compatible(&s2));
    }

    #[test]
    fn test_custom_constraint() {
        let mut cc = CustomConstraint::new(ConstraintId::new("custom1"), "allowed states");
        cc.add_allowed(vec![VersionIndex(0), VersionIndex(0)]);
        cc.add_allowed(vec![VersionIndex(1), VersionIndex(1)]);

        let s1 = ClusterState::new(&[VersionIndex(0), VersionIndex(0)]);
        assert!(cc.evaluate(&s1).status.is_satisfied());

        let s2 = ClusterState::new(&[VersionIndex(0), VersionIndex(1)]);
        assert!(cc.evaluate(&s2).status.is_violated());
    }

    #[test]
    fn test_constraint_set_merge() {
        let mut cs1 = ConstraintSet::new();
        cs1.add(Constraint::Custom(CustomConstraint::new(
            ConstraintId::new("a"),
            "first",
        )));
        let mut cs2 = ConstraintSet::new();
        cs2.add(Constraint::Custom(CustomConstraint::new(
            ConstraintId::new("b"),
            "second",
        )));
        cs1.merge(cs2);
        assert_eq!(cs1.len(), 2);
    }

    #[test]
    fn test_constraint_set_total_penalty() {
        let mut cs = ConstraintSet::new();
        let mut cc = CustomConstraint::new(ConstraintId::new("soft1"), "soft");
        cc.strength = ConstraintStrength::Soft(OrderedFloat(5.0));
        cc.add_allowed(vec![VersionIndex(1)]);
        cs.add(Constraint::Custom(cc));

        let s = ClusterState::new(&[VersionIndex(0)]);
        let penalty = cs.total_penalty(&s);
        assert!((penalty - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_infer_interval_structure() {
        let mut cc = CompatibilityConstraint::new(
            ConstraintId::new("c"),
            0,
            1,
            "test",
        );
        cc.add_compatible_pair(VersionIndex(0), VersionIndex(0));
        cc.add_compatible_pair(VersionIndex(0), VersionIndex(1));
        cc.add_compatible_pair(VersionIndex(1), VersionIndex(1));
        cc.add_compatible_pair(VersionIndex(1), VersionIndex(2));

        let ic = infer_interval_structure(&cc, 2, 3);
        assert!(ic.is_some());
        let ic = ic.unwrap();
        assert!(ic.is_compatible(VersionIndex(0), VersionIndex(0)));
        assert!(ic.is_compatible(VersionIndex(0), VersionIndex(1)));
        assert!(ic.is_compatible(VersionIndex(1), VersionIndex(1)));
        assert!(ic.is_compatible(VersionIndex(1), VersionIndex(2)));
    }

    #[test]
    fn test_constraint_evaluation_display() {
        let eval = ConstraintEvaluation::violated(
            ConstraintId::new("c1"),
            ConstraintStrength::Hard,
            "bad state",
        );
        let s = eval.to_string();
        assert!(s.contains("VIOLATED"));
        assert!(s.contains("bad state"));
    }
}

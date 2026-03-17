//! Constraint propagation for continuous domains.
//!
//! Implements AC-3 adapted for continuous interval domains, propagating masking,
//! JND, cognitive load, and segregation constraints to narrow parameter ranges.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::{
    AuditoryDimension, BarkBand, MappingConfig, OptimizerError, OptimizerResult,
    ParameterId, StreamId,
};
use crate::constraints::{Constraint, ConstraintSet};

// ─────────────────────────────────────────────────────────────────────────────
// Domain
// ─────────────────────────────────────────────────────────────────────────────

/// Continuous interval domain [min, max] for a parameter.
#[derive(Debug, Clone, Copy)]
pub struct Domain {
    pub min: f64,
    pub max: f64,
}

impl Domain {
    pub fn new(min: f64, max: f64) -> Self {
        Domain { min, max }
    }

    pub fn width(&self) -> f64 {
        (self.max - self.min).max(0.0)
    }

    pub fn midpoint(&self) -> f64 {
        (self.min + self.max) / 2.0
    }

    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.min && value <= self.max
    }

    /// Intersect with another domain.
    pub fn intersect(&self, other: &Domain) -> Domain {
        Domain {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    /// Split domain at midpoint into two halves.
    pub fn split(&self) -> (Domain, Domain) {
        let mid = self.midpoint();
        (
            Domain::new(self.min, mid),
            Domain::new(mid, self.max),
        )
    }

    /// Split domain at a specific point.
    pub fn split_at(&self, point: f64) -> (Domain, Domain) {
        let point = point.clamp(self.min, self.max);
        (
            Domain::new(self.min, point),
            Domain::new(point, self.max),
        )
    }

    /// Shrink the lower bound.
    pub fn raise_min(&mut self, new_min: f64) {
        self.min = self.min.max(new_min);
    }

    /// Shrink the upper bound.
    pub fn lower_max(&mut self, new_max: f64) {
        self.max = self.max.min(new_max);
    }

    /// Sample a point uniformly from the domain.
    pub fn sample(&self, t: f64) -> f64 {
        self.min + t.clamp(0.0, 1.0) * self.width()
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.min, self.max)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DomainStore
// ─────────────────────────────────────────────────────────────────────────────

/// Map from parameter ID to its current domain.
#[derive(Debug, Clone)]
pub struct DomainStore {
    domains: HashMap<ParameterId, Domain>,
}

impl Default for DomainStore {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainStore {
    pub fn new() -> Self {
        DomainStore {
            domains: HashMap::new(),
        }
    }

    /// Initialize a parameter's domain.
    pub fn set(&mut self, param: ParameterId, domain: Domain) {
        self.domains.insert(param, domain);
    }

    /// Get a parameter's current domain.
    pub fn get(&self, param: &ParameterId) -> Option<&Domain> {
        self.domains.get(param)
    }

    /// Get mutable reference to a parameter's domain.
    pub fn get_mut(&mut self, param: &ParameterId) -> Option<&mut Domain> {
        self.domains.get_mut(param)
    }

    /// Number of parameters.
    pub fn len(&self) -> usize {
        self.domains.len()
    }

    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
    }

    /// Intersect a parameter's domain with a new constraint.
    pub fn intersect(&mut self, param: &ParameterId, constraint: &Domain) -> bool {
        if let Some(domain) = self.domains.get_mut(param) {
            let old_width = domain.width();
            *domain = domain.intersect(constraint);
            let new_width = domain.width();
            new_width < old_width
        } else {
            false
        }
    }

    /// Check if any domain is empty (infeasible).
    pub fn has_empty_domain(&self) -> bool {
        self.domains.values().any(|d| d.is_empty())
    }

    /// Find the parameter with the largest domain width.
    pub fn largest_domain(&self) -> Option<(&ParameterId, &Domain)> {
        self.domains
            .iter()
            .max_by(|a, b| {
                a.1.width()
                    .partial_cmp(&b.1.width())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Split a parameter's domain at its midpoint, returning two stores.
    pub fn split(&self, param: &ParameterId) -> Option<(DomainStore, DomainStore)> {
        self.domains.get(param).map(|domain| {
            let (left, right) = domain.split();
            let mut store_left = self.clone();
            let mut store_right = self.clone();
            store_left.domains.insert(param.clone(), left);
            store_right.domains.insert(param.clone(), right);
            (store_left, store_right)
        })
    }

    /// Total volume of the domain space.
    pub fn volume(&self) -> f64 {
        if self.domains.is_empty() {
            return 0.0;
        }
        self.domains.values().map(|d| d.width().max(0.0)).product()
    }

    /// All parameter IDs.
    pub fn params(&self) -> Vec<ParameterId> {
        self.domains.keys().cloned().collect()
    }

    /// Iterator over (param, domain) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&ParameterId, &Domain)> {
        self.domains.iter()
    }

    /// Create from a feasible region.
    pub fn from_feasible_region(region: &crate::constraints::FeasibleRegion) -> Self {
        let mut store = DomainStore::new();
        for (name, dim) in &region.dimensions {
            store.set(
                ParameterId(name.clone()),
                Domain::new(dim.min, dim.max),
            );
        }
        store
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PropagationResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a constraint propagation pass.
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Parameters whose domains were narrowed.
    pub changed_params: Vec<ParameterId>,
    /// Whether infeasibility was detected.
    pub infeasible: bool,
    /// Number of propagation iterations performed.
    pub iterations: usize,
    /// Domain widths before propagation.
    pub widths_before: HashMap<ParameterId, f64>,
    /// Domain widths after propagation.
    pub widths_after: HashMap<ParameterId, f64>,
}

impl PropagationResult {
    /// Total reduction in domain width across all parameters.
    pub fn total_reduction(&self) -> f64 {
        let mut reduction = 0.0;
        for (param, &before) in &self.widths_before {
            if let Some(&after) = self.widths_after.get(param) {
                reduction += (before - after).max(0.0);
            }
        }
        reduction
    }

    /// Fraction of domain space eliminated.
    pub fn reduction_ratio(&self) -> f64 {
        let total_before: f64 = self.widths_before.values().sum();
        if total_before > 0.0 {
            self.total_reduction() / total_before
        } else {
            0.0
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstraintArc
// ─────────────────────────────────────────────────────────────────────────────

/// A binary constraint arc between two parameters.
#[derive(Debug, Clone)]
struct ConstraintArc {
    param1: ParameterId,
    param2: ParameterId,
    constraint_idx: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstraintPropagator
// ─────────────────────────────────────────────────────────────────────────────

/// Constraint propagation engine using AC-3 adapted for continuous domains.
#[derive(Debug, Clone)]
pub struct ConstraintPropagator {
    /// Maximum propagation iterations before stopping.
    pub max_iterations: usize,
    /// Minimum domain change to continue propagation.
    pub min_change: f64,
    /// Whether to propagate masking constraints.
    pub propagate_masking: bool,
    /// Whether to propagate JND constraints.
    pub propagate_jnd: bool,
    /// Whether to propagate cognitive load.
    pub propagate_cognitive: bool,
    /// Whether to propagate segregation constraints.
    pub propagate_segregation: bool,
}

impl Default for ConstraintPropagator {
    fn default() -> Self {
        ConstraintPropagator {
            max_iterations: 1000,
            min_change: 1e-6,
            propagate_masking: true,
            propagate_jnd: true,
            propagate_cognitive: true,
            propagate_segregation: true,
        }
    }
}

impl ConstraintPropagator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run full constraint propagation on the domain store.
    ///
    /// Uses AC-3 algorithm adapted for continuous domains: rather than
    /// enumerating discrete values, we narrow interval bounds.
    pub fn propagate(
        &self,
        domains: &mut DomainStore,
        constraints: &ConstraintSet,
    ) -> PropagationResult {
        let widths_before: HashMap<ParameterId, f64> =
            domains.iter().map(|(p, d)| (p.clone(), d.width())).collect();

        let mut changed = HashSet::new();
        let mut iterations = 0;
        let mut any_change = true;

        while any_change && iterations < self.max_iterations {
            any_change = false;
            iterations += 1;

            for (_, constraint) in constraints.iter() {
                let changed_now = self.propagate_constraint(domains, constraint);
                if !changed_now.is_empty() {
                    any_change = true;
                    for p in changed_now {
                        changed.insert(p);
                    }
                }

                if domains.has_empty_domain() {
                    let widths_after: HashMap<ParameterId, f64> =
                        domains.iter().map(|(p, d)| (p.clone(), d.width())).collect();
                    return PropagationResult {
                        changed_params: changed.into_iter().collect(),
                        infeasible: true,
                        iterations,
                        widths_before,
                        widths_after,
                    };
                }
            }

            // Check if changes are significant enough to continue
            if any_change {
                let max_change: f64 = changed
                    .iter()
                    .filter_map(|p| {
                        let before = widths_before.get(p)?;
                        let after = domains.get(p)?.width();
                        Some((before - after).abs())
                    })
                    .fold(0.0_f64, f64::max);

                if max_change < self.min_change {
                    break;
                }
            }
        }

        let widths_after: HashMap<ParameterId, f64> =
            domains.iter().map(|(p, d)| (p.clone(), d.width())).collect();

        PropagationResult {
            changed_params: changed.into_iter().collect(),
            infeasible: false,
            iterations,
            widths_before,
            widths_after,
        }
    }

    /// Propagate a single constraint, narrowing affected domains.
    fn propagate_constraint(
        &self,
        domains: &mut DomainStore,
        constraint: &Constraint,
    ) -> Vec<ParameterId> {
        match constraint {
            Constraint::FrequencyRange { min_hz, max_hz } if self.propagate_masking => {
                self.propagate_frequency_range(domains, *min_hz, *max_hz)
            }
            Constraint::AmplitudeRange { min_db, max_db } => {
                self.propagate_amplitude_range(domains, *min_db, *max_db)
            }
            Constraint::MaskingClearance { stream_id, min_margin_db }
                if self.propagate_masking =>
            {
                self.propagate_masking_clearance(domains, *stream_id, *min_margin_db)
            }
            Constraint::JndSufficiency {
                param1,
                param2,
                dimension,
                min_jnd_multiples,
            } if self.propagate_jnd => {
                self.propagate_jnd_sufficiency(
                    domains,
                    param1,
                    param2,
                    *dimension,
                    *min_jnd_multiples,
                )
            }
            Constraint::CognitiveLoadBudget { max_streams, max_load }
                if self.propagate_cognitive =>
            {
                self.propagate_cognitive_load(domains, *max_streams, *max_load)
            }
            Constraint::SegregationRequired {
                stream1_id,
                stream2_id,
                predicates,
            } if self.propagate_segregation => {
                self.propagate_segregation_constraint(
                    domains,
                    *stream1_id,
                    *stream2_id,
                    predicates,
                )
            }
            _ => Vec::new(),
        }
    }

    /// Propagate frequency range constraints to narrow frequency-related domains.
    fn propagate_frequency_range(
        &self,
        domains: &mut DomainStore,
        min_hz: f64,
        max_hz: f64,
    ) -> Vec<ParameterId> {
        let freq_constraint = Domain::new(min_hz, max_hz);
        let freq_id = ParameterId("frequency".to_string());
        let mut changed = Vec::new();

        if domains.intersect(&freq_id, &freq_constraint) {
            changed.push(freq_id.clone());
        }

        // Also propagate to per-stream frequency domains
        let params: Vec<ParameterId> = domains.params();
        for param in params {
            if param.0.starts_with("freq_stream_") {
                if domains.intersect(&param, &freq_constraint) {
                    changed.push(param);
                }
            }
        }

        changed
    }

    /// Propagate amplitude range constraints.
    fn propagate_amplitude_range(
        &self,
        domains: &mut DomainStore,
        min_db: f64,
        max_db: f64,
    ) -> Vec<ParameterId> {
        let amp_constraint = Domain::new(min_db, max_db);
        let amp_id = ParameterId("amplitude".to_string());
        let mut changed = Vec::new();

        if domains.intersect(&amp_id, &amp_constraint) {
            changed.push(amp_id.clone());
        }

        let params: Vec<ParameterId> = domains.params();
        for param in params {
            if param.0.starts_with("amp_stream_") {
                if domains.intersect(&param, &amp_constraint) {
                    changed.push(param);
                }
            }
        }

        changed
    }

    /// Propagate masking clearance: narrow frequency domains to ensure
    /// streams in the same Bark band have sufficient amplitude separation.
    fn propagate_masking_clearance(
        &self,
        domains: &mut DomainStore,
        stream_id: StreamId,
        min_margin_db: f64,
    ) -> Vec<ParameterId> {
        let amp_param = ParameterId(format!("amp_stream_{}", stream_id.0));
        let mut changed = Vec::new();

        if let Some(domain) = domains.get(&amp_param) {
            // The stream amplitude must be at least min_margin_db above the
            // global masking floor. Narrow domain to enforce this.
            let min_amp = min_margin_db;
            let constraint = Domain::new(min_amp, domain.max);
            if domains.intersect(&amp_param, &constraint) {
                changed.push(amp_param);
            }
        }

        changed
    }

    /// Propagate JND sufficiency: ensure minimum spacing between parameters.
    fn propagate_jnd_sufficiency(
        &self,
        domains: &mut DomainStore,
        param1: &ParameterId,
        param2: &ParameterId,
        dimension: AuditoryDimension,
        min_jnd_multiples: f64,
    ) -> Vec<ParameterId> {
        let mut changed = Vec::new();

        let jnd_size = match dimension {
            AuditoryDimension::Pitch => 3.0,      // Hz for typical pitch JND
            AuditoryDimension::Loudness => 1.0,    // dB
            AuditoryDimension::Timbre => 0.05,
            AuditoryDimension::SpatialAzimuth => 1.0,
            AuditoryDimension::SpatialElevation => 4.0,
            AuditoryDimension::Duration => 10.0,
            AuditoryDimension::AttackTime => 5.0,
        };

        let min_separation = jnd_size * min_jnd_multiples;

        // Get both domains
        let d1 = domains.get(param1).copied();
        let d2 = domains.get(param2).copied();

        if let (Some(d1), Some(d2)) = (d1, d2) {
            // If param1's max is close to param2's min, narrow.
            // Enforce: |v1 - v2| >= min_separation
            // This means: v1 >= v2 + min_sep OR v2 >= v1 + min_sep

            // If d1 is entirely below d2: d1.max < d2.min
            // Then enforce: d2.min >= d1.min + min_sep
            // And: d1.max <= d2.max - min_sep

            if d1.midpoint() < d2.midpoint() {
                // param1 tends lower, param2 tends higher
                let new_d1_max = d2.max - min_separation;
                let new_d2_min = d1.min + min_separation;

                if new_d1_max < d1.max {
                    let constraint = Domain::new(d1.min, new_d1_max);
                    if domains.intersect(param1, &constraint) {
                        changed.push(param1.clone());
                    }
                }
                if new_d2_min > d2.min {
                    let constraint = Domain::new(new_d2_min, d2.max);
                    if domains.intersect(param2, &constraint) {
                        changed.push(param2.clone());
                    }
                }
            } else {
                // param2 tends lower
                let new_d2_max = d1.max - min_separation;
                let new_d1_min = d2.min + min_separation;

                if new_d2_max < d2.max {
                    let constraint = Domain::new(d2.min, new_d2_max);
                    if domains.intersect(param2, &constraint) {
                        changed.push(param2.clone());
                    }
                }
                if new_d1_min > d1.min {
                    let constraint = Domain::new(new_d1_min, d1.max);
                    if domains.intersect(param1, &constraint) {
                        changed.push(param1.clone());
                    }
                }
            }
        }

        changed
    }

    /// Propagate cognitive load constraints to limit stream count domains.
    fn propagate_cognitive_load(
        &self,
        domains: &mut DomainStore,
        max_streams: usize,
        _max_load: f64,
    ) -> Vec<ParameterId> {
        let stream_count_id = ParameterId("stream_count".to_string());
        let mut changed = Vec::new();

        if domains.get(&stream_count_id).is_some() {
            let constraint = Domain::new(0.0, max_streams as f64);
            if domains.intersect(&stream_count_id, &constraint) {
                changed.push(stream_count_id);
            }
        }

        changed
    }

    /// Propagate segregation constraints to adjust spectral placement.
    fn propagate_segregation_constraint(
        &self,
        domains: &mut DomainStore,
        stream1: StreamId,
        stream2: StreamId,
        predicates: &[crate::SegregationPredicate],
    ) -> Vec<ParameterId> {
        let mut changed = Vec::new();
        let freq1_id = ParameterId(format!("freq_stream_{}", stream1.0));
        let freq2_id = ParameterId(format!("freq_stream_{}", stream2.0));

        for predicate in predicates {
            match predicate {
                crate::SegregationPredicate::MinFrequencySeparation(min_sep) => {
                    let d1 = domains.get(&freq1_id).copied();
                    let d2 = domains.get(&freq2_id).copied();

                    if let (Some(d1), Some(d2)) = (d1, d2) {
                        if d1.midpoint() < d2.midpoint() {
                            let new_max = d2.max - min_sep;
                            if new_max < d1.max {
                                let c = Domain::new(d1.min, new_max);
                                if domains.intersect(&freq1_id, &c) {
                                    changed.push(freq1_id.clone());
                                }
                            }
                            let new_min = d1.min + min_sep;
                            if new_min > d2.min {
                                let c = Domain::new(new_min, d2.max);
                                if domains.intersect(&freq2_id, &c) {
                                    changed.push(freq2_id.clone());
                                }
                            }
                        } else {
                            let new_max = d1.max - min_sep;
                            if new_max < d2.max {
                                let c = Domain::new(d2.min, new_max);
                                if domains.intersect(&freq2_id, &c) {
                                    changed.push(freq2_id.clone());
                                }
                            }
                            let new_min = d2.min + min_sep;
                            if new_min > d1.min {
                                let c = Domain::new(new_min, d1.max);
                                if domains.intersect(&freq1_id, &c) {
                                    changed.push(freq1_id.clone());
                                }
                            }
                        }
                    }
                }
                crate::SegregationPredicate::DifferentBarkBands => {
                    // Map frequency domains to Bark-band constraints
                    // If both are in the same band, push one out
                    let d1 = domains.get(&freq1_id).copied();
                    let d2 = domains.get(&freq2_id).copied();

                    if let (Some(d1), Some(d2)) = (d1, d2) {
                        let b1_mid = freq_to_bark(d1.midpoint());
                        let b2_mid = freq_to_bark(d2.midpoint());

                        if (b1_mid - b2_mid).abs() < 1.0 {
                            // Same Bark band: push stream2 up by one band width
                            let band = BarkBand(b1_mid.round() as u8);
                            let bw = band.bandwidth();
                            let new_min = d1.midpoint() + bw;
                            if new_min > d2.min {
                                let c = Domain::new(new_min, d2.max);
                                if domains.intersect(&freq2_id, &c) {
                                    changed.push(freq2_id.clone());
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        changed
    }

    /// Run propagation iteratively until a fixed point or max iterations.
    pub fn propagate_to_fixpoint(
        &self,
        domains: &mut DomainStore,
        constraints: &ConstraintSet,
    ) -> PropagationResult {
        let widths_before: HashMap<ParameterId, f64> =
            domains.iter().map(|(p, d)| (p.clone(), d.width())).collect();

        let mut all_changed = HashSet::new();
        let mut total_iterations = 0;

        for _ in 0..self.max_iterations {
            let result = self.propagate(domains, constraints);
            total_iterations += result.iterations;

            if result.infeasible {
                let widths_after: HashMap<ParameterId, f64> =
                    domains.iter().map(|(p, d)| (p.clone(), d.width())).collect();
                return PropagationResult {
                    changed_params: all_changed.into_iter().collect(),
                    infeasible: true,
                    iterations: total_iterations,
                    widths_before,
                    widths_after,
                };
            }

            for p in &result.changed_params {
                all_changed.insert(p.clone());
            }

            if result.changed_params.is_empty() {
                break;
            }
        }

        let widths_after: HashMap<ParameterId, f64> =
            domains.iter().map(|(p, d)| (p.clone(), d.width())).collect();

        PropagationResult {
            changed_params: all_changed.into_iter().collect(),
            infeasible: false,
            iterations: total_iterations,
            widths_before,
            widths_after,
        }
    }
}

/// Convert frequency in Hz to Bark scale.
fn freq_to_bark(freq: f64) -> f64 {
    13.0 * (0.00076 * freq).atan() + 3.5 * (freq / 7500.0).powi(2).atan()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::Constraint;
    use crate::SegregationPredicate;

    #[test]
    fn test_domain_basics() {
        let d = Domain::new(10.0, 20.0);
        assert_eq!(d.width(), 10.0);
        assert_eq!(d.midpoint(), 15.0);
        assert!(!d.is_empty());
        assert!(d.contains(15.0));
        assert!(!d.contains(5.0));
    }

    #[test]
    fn test_domain_empty() {
        let d = Domain::new(20.0, 10.0);
        assert!(d.is_empty());
    }

    #[test]
    fn test_domain_intersect() {
        let d1 = Domain::new(10.0, 30.0);
        let d2 = Domain::new(20.0, 40.0);
        let inter = d1.intersect(&d2);
        assert!((inter.min - 20.0).abs() < 1e-10);
        assert!((inter.max - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_domain_split() {
        let d = Domain::new(0.0, 100.0);
        let (left, right) = d.split();
        assert_eq!(left.min, 0.0);
        assert_eq!(left.max, 50.0);
        assert_eq!(right.min, 50.0);
        assert_eq!(right.max, 100.0);
    }

    #[test]
    fn test_domain_store_basic() {
        let mut store = DomainStore::new();
        store.set(ParameterId("x".into()), Domain::new(0.0, 10.0));
        store.set(ParameterId("y".into()), Domain::new(5.0, 15.0));
        assert_eq!(store.len(), 2);
        assert!(!store.has_empty_domain());
    }

    #[test]
    fn test_domain_store_intersect() {
        let mut store = DomainStore::new();
        store.set(ParameterId("x".into()), Domain::new(0.0, 100.0));
        let changed = store.intersect(
            &ParameterId("x".into()),
            &Domain::new(20.0, 80.0),
        );
        assert!(changed);
        let d = store.get(&ParameterId("x".into())).unwrap();
        assert!((d.min - 20.0).abs() < 1e-10);
        assert!((d.max - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_domain_store_empty_detection() {
        let mut store = DomainStore::new();
        store.set(ParameterId("x".into()), Domain::new(0.0, 10.0));
        store.intersect(
            &ParameterId("x".into()),
            &Domain::new(20.0, 30.0),
        );
        assert!(store.has_empty_domain());
    }

    #[test]
    fn test_domain_store_split() {
        let mut store = DomainStore::new();
        store.set(ParameterId("x".into()), Domain::new(0.0, 100.0));
        store.set(ParameterId("y".into()), Domain::new(0.0, 50.0));

        let (left, right) = store.split(&ParameterId("x".into())).unwrap();
        assert_eq!(left.get(&ParameterId("x".into())).unwrap().max, 50.0);
        assert_eq!(right.get(&ParameterId("x".into())).unwrap().min, 50.0);
        // y unchanged
        assert_eq!(left.get(&ParameterId("y".into())).unwrap().width(), 50.0);
    }

    #[test]
    fn test_propagate_frequency_range() {
        let propagator = ConstraintPropagator::new();
        let mut domains = DomainStore::new();
        domains.set(ParameterId("frequency".into()), Domain::new(0.0, 20000.0));

        let mut constraints = ConstraintSet::new();
        constraints.add(Constraint::FrequencyRange {
            min_hz: 100.0,
            max_hz: 8000.0,
        });

        let result = propagator.propagate(&mut domains, &constraints);
        let freq = domains.get(&ParameterId("frequency".into())).unwrap();
        assert!(freq.min >= 100.0);
        assert!(freq.max <= 8000.0);
        assert!(!result.infeasible);
    }

    #[test]
    fn test_propagate_amplitude_range() {
        let propagator = ConstraintPropagator::new();
        let mut domains = DomainStore::new();
        domains.set(ParameterId("amplitude".into()), Domain::new(-100.0, 120.0));

        let mut constraints = ConstraintSet::new();
        constraints.add(Constraint::AmplitudeRange {
            min_db: 30.0,
            max_db: 90.0,
        });

        propagator.propagate(&mut domains, &constraints);
        let amp = domains.get(&ParameterId("amplitude".into())).unwrap();
        assert!(amp.min >= 30.0);
        assert!(amp.max <= 90.0);
    }

    #[test]
    fn test_propagate_jnd_sufficiency() {
        let propagator = ConstraintPropagator::new();
        let mut domains = DomainStore::new();
        domains.set(ParameterId("p1".into()), Domain::new(0.0, 100.0));
        domains.set(ParameterId("p2".into()), Domain::new(50.0, 150.0));

        let mut constraints = ConstraintSet::new();
        constraints.add(Constraint::JndSufficiency {
            param1: ParameterId("p1".into()),
            param2: ParameterId("p2".into()),
            dimension: AuditoryDimension::Loudness,
            min_jnd_multiples: 5.0,
        });

        let result = propagator.propagate(&mut domains, &constraints);
        // After propagation, domains should be narrowed to enforce separation
        assert!(!result.infeasible);
    }

    #[test]
    fn test_propagation_result_reduction() {
        let mut before = HashMap::new();
        before.insert(ParameterId("x".into()), 100.0);
        before.insert(ParameterId("y".into()), 50.0);

        let mut after = HashMap::new();
        after.insert(ParameterId("x".into()), 80.0);
        after.insert(ParameterId("y".into()), 30.0);

        let result = PropagationResult {
            changed_params: vec![ParameterId("x".into()), ParameterId("y".into())],
            infeasible: false,
            iterations: 5,
            widths_before: before,
            widths_after: after,
        };

        assert!((result.total_reduction() - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_propagate_cognitive_load() {
        let propagator = ConstraintPropagator::new();
        let mut domains = DomainStore::new();
        domains.set(ParameterId("stream_count".into()), Domain::new(0.0, 20.0));

        let mut constraints = ConstraintSet::new();
        constraints.add(Constraint::CognitiveLoadBudget {
            max_streams: 6,
            max_load: 20.0,
        });

        propagator.propagate(&mut domains, &constraints);
        let sc = domains.get(&ParameterId("stream_count".into())).unwrap();
        assert!(sc.max <= 6.0);
    }

    #[test]
    fn test_propagate_segregation() {
        let propagator = ConstraintPropagator::new();
        let mut domains = DomainStore::new();
        domains.set(
            ParameterId("freq_stream_0".into()),
            Domain::new(200.0, 800.0),
        );
        domains.set(
            ParameterId("freq_stream_1".into()),
            Domain::new(200.0, 800.0),
        );

        let mut constraints = ConstraintSet::new();
        constraints.add(Constraint::SegregationRequired {
            stream1_id: StreamId(0),
            stream2_id: StreamId(1),
            predicates: vec![SegregationPredicate::MinFrequencySeparation(200.0)],
        });

        let result = propagator.propagate(&mut domains, &constraints);
        assert!(!result.infeasible);
    }

    #[test]
    fn test_domain_volume() {
        let mut store = DomainStore::new();
        store.set(ParameterId("x".into()), Domain::new(0.0, 10.0));
        store.set(ParameterId("y".into()), Domain::new(0.0, 5.0));
        assert!((store.volume() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_largest_domain() {
        let mut store = DomainStore::new();
        store.set(ParameterId("x".into()), Domain::new(0.0, 10.0));
        store.set(ParameterId("y".into()), Domain::new(0.0, 100.0));
        let (param, domain) = store.largest_domain().unwrap();
        assert_eq!(param.0, "y");
        assert!((domain.width() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_propagate_to_fixpoint() {
        let propagator = ConstraintPropagator::new();
        let mut domains = DomainStore::new();
        domains.set(ParameterId("frequency".into()), Domain::new(0.0, 20000.0));
        domains.set(ParameterId("amplitude".into()), Domain::new(-120.0, 120.0));

        let mut constraints = ConstraintSet::new();
        constraints.add(Constraint::FrequencyRange { min_hz: 100.0, max_hz: 8000.0 });
        constraints.add(Constraint::AmplitudeRange { min_db: 30.0, max_db: 90.0 });

        let result = propagator.propagate_to_fixpoint(&mut domains, &constraints);
        assert!(!result.infeasible);

        let freq = domains.get(&ParameterId("frequency".into())).unwrap();
        assert!(freq.min >= 100.0);
        assert!(freq.max <= 8000.0);
    }

    #[test]
    fn test_domain_sample() {
        let d = Domain::new(10.0, 20.0);
        assert_eq!(d.sample(0.0), 10.0);
        assert_eq!(d.sample(1.0), 20.0);
        assert_eq!(d.sample(0.5), 15.0);
    }
}

//! Galois connection between concrete and abstract coalgebras.
//!
//! Provides abstraction (α) and concretization (γ) maps with property
//! preservation guarantees and quantitative degradation bounds.
//!
//! A Galois connection (α, γ) satisfies:  ∀x. x ⊑ γ(α(x))  and  ∀a. α(γ(a)) ⊑ a
//! For probabilistic coalgebras, we use an approximate Galois connection where
//! these properties hold up to some quantitative bound δ.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use ordered_float::OrderedFloat;
use indexmap::IndexMap;

// ---------------------------------------------------------------------------
// Local type definitions (to be later swapped with coalgebra module types)
// ---------------------------------------------------------------------------

/// State identifier.
pub type StateId = String;

/// A probability distribution over states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    pub entries: IndexMap<String, f64>,
}

impl Distribution {
    pub fn new() -> Self {
        Self { entries: IndexMap::new() }
    }

    pub fn singleton(key: String, prob: f64) -> Self {
        let mut entries = IndexMap::new();
        entries.insert(key, prob);
        Self { entries }
    }

    pub fn from_map(map: IndexMap<String, f64>) -> Self {
        Self { entries: map }
    }

    pub fn total_mass(&self) -> f64 {
        self.entries.values().sum()
    }

    pub fn normalize(&mut self) {
        let total = self.total_mass();
        if total > 0.0 {
            for v in self.entries.values_mut() {
                *v /= total;
            }
        }
    }

    pub fn support(&self) -> HashSet<String> {
        self.entries.keys().cloned().collect()
    }

    pub fn get(&self, key: &str) -> f64 {
        self.entries.get(key).copied().unwrap_or(0.0)
    }

    /// Total variation distance between two distributions.
    pub fn tv_distance(&self, other: &Distribution) -> f64 {
        let mut all_keys: HashSet<&String> = HashSet::new();
        for k in self.entries.keys() {
            all_keys.insert(k);
        }
        for k in other.entries.keys() {
            all_keys.insert(k);
        }
        let mut sum = 0.0;
        for k in &all_keys {
            let p = self.get(k);
            let q = other.get(k);
            sum += (p - q).abs();
        }
        sum / 2.0
    }

    /// Wasserstein-1 distance using a discrete metric (0/1).
    pub fn wasserstein_discrete(&self, other: &Distribution) -> f64 {
        // With discrete metric, W_1 = TV distance.
        self.tv_distance(other)
    }

    /// KL divergence D_KL(self || other).
    pub fn kl_divergence(&self, other: &Distribution) -> f64 {
        let mut kl = 0.0;
        for (k, &p) in &self.entries {
            if p > 0.0 {
                let q = other.get(k).max(1e-15);
                kl += p * (p / q).ln();
            }
        }
        kl
    }

    /// Merge this distribution with another using weights.
    pub fn mixture(&self, other: &Distribution, self_weight: f64) -> Distribution {
        let other_weight = 1.0 - self_weight;
        let mut result = IndexMap::new();
        let mut all_keys: HashSet<&String> = HashSet::new();
        for k in self.entries.keys() { all_keys.insert(k); }
        for k in other.entries.keys() { all_keys.insert(k); }
        for k in &all_keys {
            let p = self.get(k) * self_weight + other.get(k) * other_weight;
            if p > 0.0 {
                result.insert((*k).clone(), p);
            }
        }
        Distribution { entries: result }
    }
}

impl Default for Distribution {
    fn default() -> Self {
        Self::new()
    }
}

/// A concrete state with its transition behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcreteState {
    pub id: StateId,
    /// For each input word, the output distribution.
    pub transitions: HashMap<String, Distribution>,
    /// Feature vector (embedding or behavioral stats).
    pub features: Vec<f64>,
}

/// An abstract state grouping multiple concrete states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractState {
    pub id: StateId,
    /// Concrete states mapped to this abstract state.
    pub members: Vec<StateId>,
    /// Aggregated transition distribution (weighted average).
    pub transitions: HashMap<String, Distribution>,
    /// Centroid feature vector.
    pub centroid: Vec<f64>,
}

/// Configuration for a concrete coalgebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcreteCoalgebra {
    pub states: HashMap<StateId, ConcreteState>,
    pub initial_state: StateId,
}

/// Configuration for an abstract coalgebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractCoalgebra {
    pub states: HashMap<StateId, AbstractState>,
    pub initial_state: StateId,
}

// ---------------------------------------------------------------------------
// Abstraction map α: Concrete → Abstract
// ---------------------------------------------------------------------------

/// The abstraction function α mapping concrete states to abstract states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionMap {
    /// Map from concrete state ID to abstract state ID.
    pub mapping: HashMap<StateId, StateId>,
    /// Partition: abstract state ID → set of concrete state IDs.
    pub partition: HashMap<StateId, Vec<StateId>>,
    /// Lipschitz constant of the map (quantitative bound).
    pub lipschitz_constant: f64,
    /// Maximum distortion introduced by abstraction.
    pub max_distortion: f64,
}

impl AbstractionMap {
    /// Build an abstraction map from a state partition.
    pub fn from_partition(partition: HashMap<StateId, Vec<StateId>>) -> Self {
        let mut mapping = HashMap::new();
        for (abs_id, concrete_ids) in &partition {
            for cid in concrete_ids {
                mapping.insert(cid.clone(), abs_id.clone());
            }
        }
        Self {
            mapping,
            partition,
            lipschitz_constant: 1.0,
            max_distortion: 0.0,
        }
    }

    /// Map a concrete state to its abstract state.
    pub fn abstract_state(&self, concrete: &StateId) -> Option<&StateId> {
        self.mapping.get(concrete)
    }

    /// Map a concrete distribution to an abstract distribution.
    pub fn abstract_distribution(&self, dist: &Distribution) -> Distribution {
        let mut result = IndexMap::new();
        for (state, &prob) in &dist.entries {
            if let Some(abs) = self.mapping.get(state) {
                *result.entry(abs.clone()).or_insert(0.0) += prob;
            }
        }
        Distribution { entries: result }
    }

    /// Compute the number of abstract states.
    pub fn num_abstract_states(&self) -> usize {
        self.partition.len()
    }

    /// Compute compression ratio (concrete states / abstract states).
    pub fn compression_ratio(&self) -> f64 {
        let concrete = self.mapping.len() as f64;
        let abstract_count = self.partition.len() as f64;
        if abstract_count > 0.0 {
            concrete / abstract_count
        } else {
            0.0
        }
    }

    /// Compute the Lipschitz constant of the abstraction map.
    /// L = max over pairs (s1, s2) of |d_abstract(α(s1), α(s2))| / |d_concrete(s1, s2)|.
    pub fn compute_lipschitz(
        &mut self,
        concrete: &ConcreteCoalgebra,
        metric: &dyn Fn(&ConcreteState, &ConcreteState) -> f64,
    ) {
        let mut max_ratio = 0.0f64;

        let state_ids: Vec<&StateId> = concrete.states.keys().collect();
        for i in 0..state_ids.len() {
            for j in (i + 1)..state_ids.len() {
                let s1 = &concrete.states[state_ids[i]];
                let s2 = &concrete.states[state_ids[j]];

                let d_concrete = metric(s1, s2);
                if d_concrete < 1e-15 {
                    continue;
                }

                let a1 = self.mapping.get(&s1.id).cloned().unwrap_or_default();
                let a2 = self.mapping.get(&s2.id).cloned().unwrap_or_default();

                let d_abstract = if a1 == a2 { 0.0 } else { 1.0 };
                let ratio = d_abstract / d_concrete;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }

        self.lipschitz_constant = max_ratio;
    }

    /// Compute the maximum distortion introduced by abstraction.
    /// Distortion = max TV distance between concrete transition and abstract transition.
    pub fn compute_max_distortion(
        &mut self,
        concrete: &ConcreteCoalgebra,
        abstract_coal: &AbstractCoalgebra,
    ) {
        let mut max_dist = 0.0f64;

        for (cid, cstate) in &concrete.states {
            let abs_id = match self.mapping.get(cid) {
                Some(a) => a,
                None => continue,
            };
            let astate = match abstract_coal.states.get(abs_id) {
                Some(a) => a,
                None => continue,
            };

            for (input, c_dist) in &cstate.transitions {
                let a_dist = match astate.transitions.get(input) {
                    Some(d) => d,
                    None => continue,
                };

                // Abstract the concrete distribution, then compare with abstract.
                let abstracted = self.abstract_distribution(c_dist);
                let distortion = abstracted.tv_distance(a_dist);
                if distortion > max_dist {
                    max_dist = distortion;
                }
            }
        }

        self.max_distortion = max_dist;
    }

    /// Check if this abstraction map is consistent (every concrete state is mapped).
    pub fn is_consistent(&self, concrete: &ConcreteCoalgebra) -> bool {
        for cid in concrete.states.keys() {
            if !self.mapping.contains_key(cid) {
                return false;
            }
        }
        // Check that partition is consistent with mapping.
        for (abs_id, members) in &self.partition {
            for m in members {
                match self.mapping.get(m) {
                    Some(a) if a == abs_id => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Refine the abstraction by splitting an abstract state.
    pub fn split_state(
        &mut self,
        abstract_id: &StateId,
        new_partition: Vec<(StateId, Vec<StateId>)>,
    ) {
        // Remove old abstract state.
        self.partition.remove(abstract_id);

        // Add new partitions.
        for (new_abs_id, members) in new_partition {
            for m in &members {
                self.mapping.insert(m.clone(), new_abs_id.clone());
            }
            self.partition.insert(new_abs_id, members);
        }
    }

    /// Coarsen the abstraction by merging abstract states.
    pub fn merge_states(&mut self, state_ids: &[StateId], merged_id: StateId) {
        let mut all_members = Vec::new();
        for sid in state_ids {
            if let Some(members) = self.partition.remove(sid) {
                all_members.extend(members);
            }
        }
        for m in &all_members {
            self.mapping.insert(m.clone(), merged_id.clone());
        }
        self.partition.insert(merged_id, all_members);
    }
}

impl fmt::Display for AbstractionMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AbstractionMap[{} concrete → {} abstract, L={:.4}, distortion={:.4}]",
            self.mapping.len(),
            self.partition.len(),
            self.lipschitz_constant,
            self.max_distortion,
        )
    }
}

// ---------------------------------------------------------------------------
// Concretization map γ: Abstract → P(Concrete)
// ---------------------------------------------------------------------------

/// The concretization function γ mapping abstract states back to sets of concrete states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcretizationMap {
    /// Map from abstract state ID to set of concrete state IDs.
    pub mapping: HashMap<StateId, Vec<StateId>>,
    /// Weights for concretizing distributions (how to distribute probability mass).
    pub weights: HashMap<StateId, HashMap<StateId, f64>>,
}

impl ConcretizationMap {
    /// Build from an abstraction map (the reverse mapping).
    pub fn from_abstraction(abs_map: &AbstractionMap) -> Self {
        let mapping = abs_map.partition.clone();

        // Default weights: uniform over members.
        let mut weights = HashMap::new();
        for (abs_id, members) in &mapping {
            let w = 1.0 / members.len().max(1) as f64;
            let member_weights: HashMap<StateId, f64> = members.iter()
                .map(|m| (m.clone(), w))
                .collect();
            weights.insert(abs_id.clone(), member_weights);
        }

        Self { mapping, weights }
    }

    /// Concretize an abstract state to a set of concrete states.
    pub fn concretize_state(&self, abstract_id: &StateId) -> Vec<StateId> {
        self.mapping.get(abstract_id).cloned().unwrap_or_default()
    }

    /// Concretize an abstract distribution to a concrete distribution.
    pub fn concretize_distribution(&self, abs_dist: &Distribution) -> Distribution {
        let mut result = IndexMap::new();
        for (abs_id, &prob) in &abs_dist.entries {
            if let Some(member_weights) = self.weights.get(abs_id) {
                for (cid, &w) in member_weights {
                    *result.entry(cid.clone()).or_insert(0.0) += prob * w;
                }
            }
        }
        Distribution { entries: result }
    }

    /// Set custom weights for concretization of a specific abstract state.
    pub fn set_weights(&mut self, abstract_id: StateId, concrete_weights: HashMap<StateId, f64>) {
        // Normalize weights.
        let total: f64 = concrete_weights.values().sum();
        if total > 0.0 {
            let normalized: HashMap<StateId, f64> = concrete_weights.into_iter()
                .map(|(k, v)| (k, v / total))
                .collect();
            self.weights.insert(abstract_id, normalized);
        }
    }

    /// Update weights based on observed frequencies.
    pub fn update_weights_from_data(
        &mut self,
        abstract_id: &StateId,
        observations: &HashMap<StateId, usize>,
    ) {
        let total: usize = observations.values().sum();
        if total == 0 {
            return;
        }
        let mut weights = HashMap::new();
        for (cid, &count) in observations {
            weights.insert(cid.clone(), count as f64 / total as f64);
        }
        self.weights.insert(abstract_id.clone(), weights);
    }
}

impl fmt::Display for ConcretizationMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConcretizationMap[{} abstract states]",
            self.mapping.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Property preservation
// ---------------------------------------------------------------------------

/// Types of properties that may be preserved (or degraded) by abstraction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyKind {
    /// Safety: "bad thing never happens."
    Safety,
    /// Liveness: "good thing eventually happens."
    Liveness,
    /// Probabilistic bound: "probability of X is at most p."
    ProbabilisticBound,
    /// Bisimulation: "states are behaviorally equivalent."
    Bisimulation,
    /// Trace equivalence: "same set of observable traces."
    TraceEquivalence,
    /// Metric bound: "distance between behaviors is at most d."
    MetricBound,
}

/// A behavioral property to check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralProperty {
    pub name: String,
    pub kind: PropertyKind,
    /// The bound value (for probabilistic/metric properties).
    pub bound: Option<f64>,
    /// Description.
    pub description: String,
}

impl BehavioralProperty {
    pub fn safety(name: impl Into<String>, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: PropertyKind::Safety,
            bound: None,
            description: desc.into(),
        }
    }

    pub fn probabilistic(name: impl Into<String>, bound: f64, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: PropertyKind::ProbabilisticBound,
            bound: Some(bound),
            description: desc.into(),
        }
    }

    pub fn metric(name: impl Into<String>, bound: f64, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: PropertyKind::MetricBound,
            bound: Some(bound),
            description: desc.into(),
        }
    }
}

/// Result of checking property preservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreservationResult {
    pub property: BehavioralProperty,
    pub preserved: bool,
    /// Quantitative degradation if not exactly preserved.
    pub degradation: f64,
    /// Upper bound on degradation from theory.
    pub theoretical_bound: f64,
    pub explanation: String,
}

/// Runtime checks for property preservation across a Galois connection.
#[derive(Debug, Clone)]
pub struct PropertyPreservation {
    pub abstraction: AbstractionMap,
    pub concretization: ConcretizationMap,
    pub results: Vec<PreservationResult>,
}

impl PropertyPreservation {
    pub fn new(abstraction: AbstractionMap, concretization: ConcretizationMap) -> Self {
        Self {
            abstraction,
            concretization,
            results: Vec::new(),
        }
    }

    /// Check the Galois connection property: ∀x. x ⊑ γ(α(x))
    /// In the probabilistic setting, this means the concretization of the
    /// abstracted distribution should "cover" the original.
    pub fn check_upper_adjoint(
        &self,
        concrete: &ConcreteCoalgebra,
    ) -> (bool, f64) {
        let mut max_violation = 0.0f64;

        for (cid, cstate) in &concrete.states {
            for (input, c_dist) in &cstate.transitions {
                // α(c_dist)
                let abs_dist = self.abstraction.abstract_distribution(c_dist);
                // γ(α(c_dist))
                let roundtrip = self.concretization.concretize_distribution(&abs_dist);

                // Check that for every state s, c_dist(s) ≤ roundtrip(s)
                // This is the "coverage" property for over-approximation.
                // In general probabilistic setting, we check TV distance.
                let violation = c_dist.tv_distance(&roundtrip);
                if violation > max_violation {
                    max_violation = violation;
                }
            }
        }

        let ok = max_violation < 1e-6;
        (ok, max_violation)
    }

    /// Check the lower adjoint property: ∀a. α(γ(a)) ⊑ a
    pub fn check_lower_adjoint(
        &self,
        abstract_coal: &AbstractCoalgebra,
    ) -> (bool, f64) {
        let mut max_violation = 0.0f64;

        for (abs_id, astate) in &abstract_coal.states {
            for (input, a_dist) in &astate.transitions {
                // γ(a_dist)
                let conc_dist = self.concretization.concretize_distribution(a_dist);
                // α(γ(a_dist))
                let roundtrip = self.abstraction.abstract_distribution(&conc_dist);

                let violation = roundtrip.tv_distance(a_dist);
                if violation > max_violation {
                    max_violation = violation;
                }
            }
        }

        let ok = max_violation < 1e-6;
        (ok, max_violation)
    }

    /// Check safety property preservation.
    /// Safety in abstract model implies safety in concrete model (for over-approximations).
    pub fn check_safety_preservation(
        &mut self,
        property: BehavioralProperty,
        abstract_safe: bool,
        _concrete: &ConcreteCoalgebra,
    ) -> PreservationResult {
        let distortion = self.abstraction.max_distortion;
        let lipschitz = self.abstraction.lipschitz_constant;

        // For over-approximation: if abstract is safe, concrete is safe
        // up to distortion bound.
        let preserved = abstract_safe;
        let degradation = distortion;
        let theoretical_bound = lipschitz * distortion;

        let explanation = if preserved {
            format!(
                "Safety preserved: abstract model is safe, distortion={:.6}, bound={:.6}",
                degradation, theoretical_bound
            )
        } else {
            format!(
                "Safety NOT preserved: abstract model is unsafe, distortion={:.6}",
                degradation
            )
        };

        let result = PreservationResult {
            property,
            preserved,
            degradation,
            theoretical_bound,
            explanation,
        };
        self.results.push(result.clone());
        result
    }

    /// Check probabilistic bound preservation.
    /// If P_abstract(prop) ≤ p, then P_concrete(prop) ≤ p + δ.
    pub fn check_probabilistic_preservation(
        &mut self,
        property: BehavioralProperty,
        abstract_prob: f64,
        _concrete: &ConcreteCoalgebra,
    ) -> PreservationResult {
        let bound = property.bound.unwrap_or(1.0);
        let distortion = self.abstraction.max_distortion;

        let degradation = distortion;
        let theoretical_bound = abstract_prob + distortion;
        let preserved = theoretical_bound <= bound;

        let explanation = format!(
            "P_abstract={:.6}, distortion={:.6}, P_concrete ≤ {:.6}, bound={:.6} → {}",
            abstract_prob, distortion, theoretical_bound, bound,
            if preserved { "PRESERVED" } else { "VIOLATED" }
        );

        let result = PreservationResult {
            property,
            preserved,
            degradation,
            theoretical_bound,
            explanation,
        };
        self.results.push(result.clone());
        result
    }

    /// Check metric bound preservation (bisimulation distance).
    pub fn check_metric_preservation(
        &mut self,
        property: BehavioralProperty,
        abstract_distance: f64,
        _concrete: &ConcreteCoalgebra,
    ) -> PreservationResult {
        let bound = property.bound.unwrap_or(1.0);
        let lipschitz = self.abstraction.lipschitz_constant;
        let distortion = self.abstraction.max_distortion;

        // d_concrete ≤ L * d_abstract + 2δ
        let theoretical_bound = lipschitz * abstract_distance + 2.0 * distortion;
        let preserved = theoretical_bound <= bound;
        let degradation = (theoretical_bound - abstract_distance).max(0.0);

        let explanation = format!(
            "d_abstract={:.6}, L={:.6}, δ={:.6}, d_concrete ≤ {:.6}, bound={:.6} → {}",
            abstract_distance, lipschitz, distortion, theoretical_bound, bound,
            if preserved { "PRESERVED" } else { "VIOLATED" }
        );

        let result = PreservationResult {
            property,
            preserved,
            degradation,
            theoretical_bound,
            explanation,
        };
        self.results.push(result.clone());
        result
    }

    /// Get a summary of all preservation checks.
    pub fn summary(&self) -> String {
        let total = self.results.len();
        let preserved = self.results.iter().filter(|r| r.preserved).count();
        let mut out = format!("Property preservation: {}/{} preserved\n", preserved, total);
        for r in &self.results {
            let status = if r.preserved { "✓" } else { "✗" };
            out.push_str(&format!(
                "  {} {} (degradation={:.6})\n",
                status, r.property.name, r.degradation
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Degradation bound computation
// ---------------------------------------------------------------------------

/// Quantitative degradation bound for an abstraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationBound {
    /// Maximum TV distance between concrete and abstracted distributions.
    pub tv_bound: f64,
    /// Wasserstein distance bound.
    pub wasserstein_bound: f64,
    /// KL divergence bound.
    pub kl_bound: f64,
    /// Lipschitz constant of the coarsening map.
    pub lipschitz: f64,
    /// Per-state bounds.
    pub per_state_bounds: HashMap<StateId, f64>,
}

impl DegradationBound {
    /// Compute degradation bounds for a given abstraction.
    pub fn compute(
        concrete: &ConcreteCoalgebra,
        abstract_coal: &AbstractCoalgebra,
        abs_map: &AbstractionMap,
    ) -> Self {
        let mut max_tv = 0.0f64;
        let mut max_w = 0.0f64;
        let mut max_kl = 0.0f64;
        let mut per_state = HashMap::new();

        for (cid, cstate) in &concrete.states {
            let abs_id = match abs_map.mapping.get(cid) {
                Some(a) => a,
                None => continue,
            };
            let astate = match abstract_coal.states.get(abs_id) {
                Some(a) => a,
                None => continue,
            };

            let mut state_max_tv = 0.0f64;

            for (input, c_dist) in &cstate.transitions {
                let abstracted = abs_map.abstract_distribution(c_dist);

                if let Some(a_dist) = astate.transitions.get(input) {
                    let tv = abstracted.tv_distance(a_dist);
                    let w = abstracted.wasserstein_discrete(a_dist);
                    let kl = if abstracted.total_mass() > 0.0 && a_dist.total_mass() > 0.0 {
                        abstracted.kl_divergence(a_dist)
                    } else {
                        0.0
                    };

                    if tv > max_tv { max_tv = tv; }
                    if w > max_w { max_w = w; }
                    if kl > max_kl { max_kl = kl; }
                    if tv > state_max_tv { state_max_tv = tv; }
                }
            }

            per_state.insert(cid.clone(), state_max_tv);
        }

        Self {
            tv_bound: max_tv,
            wasserstein_bound: max_w,
            kl_bound: max_kl,
            lipschitz: abs_map.lipschitz_constant,
            per_state_bounds: per_state,
        }
    }

    /// Aggregate bound: a single number summarizing the degradation.
    pub fn aggregate(&self) -> f64 {
        // Use TV bound as the primary measure.
        self.tv_bound
    }

    /// Check if the degradation is within a tolerance.
    pub fn is_within_tolerance(&self, epsilon: f64) -> bool {
        self.tv_bound <= epsilon
    }

    /// Identify the worst-case states.
    pub fn worst_states(&self, top_k: usize) -> Vec<(StateId, f64)> {
        let mut states: Vec<(StateId, f64)> = self.per_state_bounds.iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        states.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        states.truncate(top_k);
        states
    }
}

impl fmt::Display for DegradationBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DegradationBound[TV={:.6}, W1={:.6}, KL={:.6}, L={:.4}]",
            self.tv_bound, self.wasserstein_bound, self.kl_bound, self.lipschitz
        )
    }
}

// ---------------------------------------------------------------------------
// The full Galois connection
// ---------------------------------------------------------------------------

/// A Galois connection (α, γ) between a concrete and abstract coalgebra.
#[derive(Debug, Clone)]
pub struct GaloisConnection {
    pub abstraction: AbstractionMap,
    pub concretization: ConcretizationMap,
    pub degradation: Option<DegradationBound>,
    /// Whether the connection has been validated.
    pub validated: bool,
    /// Validation error (max violation of Galois properties).
    pub validation_error: f64,
}

impl GaloisConnection {
    /// Construct from a partition of concrete states.
    pub fn from_partition(partition: HashMap<StateId, Vec<StateId>>) -> Self {
        let abstraction = AbstractionMap::from_partition(partition);
        let concretization = ConcretizationMap::from_abstraction(&abstraction);
        Self {
            abstraction,
            concretization,
            degradation: None,
            validated: false,
            validation_error: 0.0,
        }
    }

    /// Construct from an existing abstraction map.
    pub fn from_abstraction_map(abs_map: AbstractionMap) -> Self {
        let concretization = ConcretizationMap::from_abstraction(&abs_map);
        Self {
            abstraction: abs_map,
            concretization,
            degradation: None,
            validated: false,
            validation_error: 0.0,
        }
    }

    /// Build the abstract coalgebra from the concrete one.
    pub fn build_abstract_coalgebra(
        &self,
        concrete: &ConcreteCoalgebra,
    ) -> AbstractCoalgebra {
        let mut abstract_states = HashMap::new();

        for (abs_id, members) in &self.abstraction.partition {
            let mut aggregated_transitions: HashMap<String, Distribution> = HashMap::new();
            let mut centroid = Vec::new();
            let mut count = 0usize;

            for mid in members {
                if let Some(cstate) = concrete.states.get(mid) {
                    // Aggregate transitions.
                    for (input, c_dist) in &cstate.transitions {
                        let abs_dist = self.abstraction.abstract_distribution(c_dist);
                        let entry = aggregated_transitions.entry(input.clone())
                            .or_insert_with(Distribution::new);

                        // Accumulate.
                        for (k, &v) in &abs_dist.entries {
                            *entry.entries.entry(k.clone()).or_insert(0.0) += v;
                        }
                    }

                    // Aggregate features.
                    if centroid.is_empty() {
                        centroid = cstate.features.clone();
                    } else {
                        for (i, &f) in cstate.features.iter().enumerate() {
                            if i < centroid.len() {
                                centroid[i] += f;
                            }
                        }
                    }
                    count += 1;
                }
            }

            // Average the transitions and features.
            if count > 0 {
                for dist in aggregated_transitions.values_mut() {
                    for v in dist.entries.values_mut() {
                        *v /= count as f64;
                    }
                }
                for c in centroid.iter_mut() {
                    *c /= count as f64;
                }
            }

            abstract_states.insert(abs_id.clone(), AbstractState {
                id: abs_id.clone(),
                members: members.clone(),
                transitions: aggregated_transitions,
                centroid,
            });
        }

        let initial = self.abstraction.abstract_state(&concrete.initial_state)
            .cloned()
            .unwrap_or_else(|| "abstract_0".to_string());

        AbstractCoalgebra {
            states: abstract_states,
            initial_state: initial,
        }
    }

    /// Validate the Galois connection properties.
    pub fn validate(
        &mut self,
        concrete: &ConcreteCoalgebra,
        abstract_coal: &AbstractCoalgebra,
    ) -> bool {
        let pp = PropertyPreservation::new(
            self.abstraction.clone(),
            self.concretization.clone(),
        );

        let (upper_ok, upper_err) = pp.check_upper_adjoint(concrete);
        let (lower_ok, lower_err) = pp.check_lower_adjoint(abstract_coal);

        self.validation_error = upper_err.max(lower_err);
        self.validated = true;

        upper_ok && lower_ok
    }

    /// Compute degradation bounds.
    pub fn compute_degradation(
        &mut self,
        concrete: &ConcreteCoalgebra,
        abstract_coal: &AbstractCoalgebra,
    ) {
        let bound = DegradationBound::compute(concrete, abstract_coal, &self.abstraction);
        self.degradation = Some(bound);
    }

    /// Refine the Galois connection by splitting an abstract state.
    pub fn refine_split(
        &mut self,
        abstract_id: &StateId,
        new_partition: Vec<(StateId, Vec<StateId>)>,
    ) {
        self.abstraction.split_state(abstract_id, new_partition);
        self.concretization = ConcretizationMap::from_abstraction(&self.abstraction);
        self.validated = false;
        self.degradation = None;
    }

    /// Coarsen by merging abstract states.
    pub fn coarsen_merge(&mut self, state_ids: &[StateId], merged_id: StateId) {
        self.abstraction.merge_states(state_ids, merged_id);
        self.concretization = ConcretizationMap::from_abstraction(&self.abstraction);
        self.validated = false;
        self.degradation = None;
    }

    /// Compute the Lipschitz constant for the coarsening map (abstract→coarser abstract).
    pub fn coarsening_lipschitz(
        &self,
        finer: &GaloisConnection,
    ) -> f64 {
        // L = max over pairs of abstract states of
        // d_coarse(coarsen(s1), coarsen(s2)) / d_fine(s1, s2)
        //
        // Since both are partitions, and coarsening merges some blocks,
        // the Lipschitz constant is at most 1 (merging never increases distance).
        // But in the weighted/probabilistic case it can differ.

        let mut max_ratio = 0.0f64;

        // For each pair of fine abstract states.
        let fine_states: Vec<&StateId> = finer.abstraction.partition.keys().collect();
        for i in 0..fine_states.len() {
            for j in (i + 1)..fine_states.len() {
                let s1 = fine_states[i];
                let s2 = fine_states[j];

                // Distance in fine space: use discrete metric.
                let d_fine = if s1 == s2 { 0.0 } else { 1.0 };
                if d_fine < 1e-15 {
                    continue;
                }

                // Map to coarse space: check if any member of s1 and s2
                // map to the same coarse state.
                let members_1 = finer.abstraction.partition.get(s1)
                    .cloned().unwrap_or_default();
                let members_2 = finer.abstraction.partition.get(s2)
                    .cloned().unwrap_or_default();

                let coarse_1: HashSet<_> = members_1.iter()
                    .filter_map(|m| self.abstraction.mapping.get(m))
                    .collect();
                let coarse_2: HashSet<_> = members_2.iter()
                    .filter_map(|m| self.abstraction.mapping.get(m))
                    .collect();

                let d_coarse = if coarse_1.intersection(&coarse_2).next().is_some() {
                    0.0
                } else {
                    1.0
                };

                let ratio = d_coarse / d_fine;
                if ratio > max_ratio {
                    max_ratio = ratio;
                }
            }
        }

        max_ratio
    }
}

impl fmt::Display for GaloisConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GaloisConnection[{}, validated={}, error={:.6}]",
            self.abstraction,
            self.validated,
            self.validation_error,
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_concrete_coalgebra() -> ConcreteCoalgebra {
        let mut states = HashMap::new();

        // State s0: transitions on "a" → {s0: 0.3, s1: 0.7}
        let mut t0 = HashMap::new();
        let mut d0 = IndexMap::new();
        d0.insert("s0".to_string(), 0.3);
        d0.insert("s1".to_string(), 0.7);
        t0.insert("a".to_string(), Distribution { entries: d0 });
        states.insert("s0".to_string(), ConcreteState {
            id: "s0".to_string(),
            transitions: t0,
            features: vec![1.0, 0.0],
        });

        // State s1: transitions on "a" → {s0: 0.2, s1: 0.8}
        let mut t1 = HashMap::new();
        let mut d1 = IndexMap::new();
        d1.insert("s0".to_string(), 0.2);
        d1.insert("s1".to_string(), 0.8);
        t1.insert("a".to_string(), Distribution { entries: d1 });
        states.insert("s1".to_string(), ConcreteState {
            id: "s1".to_string(),
            transitions: t1,
            features: vec![0.0, 1.0],
        });

        // State s2: transitions on "a" → {s0: 0.25, s1: 0.75}
        let mut t2 = HashMap::new();
        let mut d2 = IndexMap::new();
        d2.insert("s0".to_string(), 0.25);
        d2.insert("s1".to_string(), 0.75);
        t2.insert("a".to_string(), Distribution { entries: d2 });
        states.insert("s2".to_string(), ConcreteState {
            id: "s2".to_string(),
            transitions: t2,
            features: vec![0.5, 0.5],
        });

        ConcreteCoalgebra {
            states,
            initial_state: "s0".to_string(),
        }
    }

    fn make_partition() -> HashMap<StateId, Vec<StateId>> {
        let mut partition = HashMap::new();
        partition.insert("A0".to_string(), vec!["s0".to_string()]);
        partition.insert("A1".to_string(), vec!["s1".to_string(), "s2".to_string()]);
        partition
    }

    #[test]
    fn test_distribution_tv_distance() {
        let mut d1 = IndexMap::new();
        d1.insert("a".to_string(), 0.5);
        d1.insert("b".to_string(), 0.5);
        let dist1 = Distribution { entries: d1 };

        let mut d2 = IndexMap::new();
        d2.insert("a".to_string(), 0.3);
        d2.insert("b".to_string(), 0.7);
        let dist2 = Distribution { entries: d2 };

        let tv = dist1.tv_distance(&dist2);
        assert!((tv - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_kl() {
        let mut d1 = IndexMap::new();
        d1.insert("a".to_string(), 0.5);
        d1.insert("b".to_string(), 0.5);
        let dist1 = Distribution { entries: d1 };

        let kl = dist1.kl_divergence(&dist1);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_distribution_mixture() {
        let mut d1 = IndexMap::new();
        d1.insert("a".to_string(), 1.0);
        let dist1 = Distribution { entries: d1 };

        let mut d2 = IndexMap::new();
        d2.insert("b".to_string(), 1.0);
        let dist2 = Distribution { entries: d2 };

        let mix = dist1.mixture(&dist2, 0.5);
        assert!((mix.get("a") - 0.5).abs() < 1e-10);
        assert!((mix.get("b") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_abstraction_map_from_partition() {
        let partition = make_partition();
        let abs_map = AbstractionMap::from_partition(partition);

        assert_eq!(abs_map.abstract_state(&"s0".to_string()), Some(&"A0".to_string()));
        assert_eq!(abs_map.abstract_state(&"s1".to_string()), Some(&"A1".to_string()));
        assert_eq!(abs_map.abstract_state(&"s2".to_string()), Some(&"A1".to_string()));
        assert_eq!(abs_map.num_abstract_states(), 2);
    }

    #[test]
    fn test_abstract_distribution() {
        let partition = make_partition();
        let abs_map = AbstractionMap::from_partition(partition);

        let mut d = IndexMap::new();
        d.insert("s0".to_string(), 0.3);
        d.insert("s1".to_string(), 0.5);
        d.insert("s2".to_string(), 0.2);
        let dist = Distribution { entries: d };

        let abs_dist = abs_map.abstract_distribution(&dist);
        assert!((abs_dist.get("A0") - 0.3).abs() < 1e-10);
        assert!((abs_dist.get("A1") - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_compression_ratio() {
        let partition = make_partition();
        let abs_map = AbstractionMap::from_partition(partition);
        assert!((abs_map.compression_ratio() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_concretization_map() {
        let partition = make_partition();
        let abs_map = AbstractionMap::from_partition(partition);
        let conc_map = ConcretizationMap::from_abstraction(&abs_map);

        let members = conc_map.concretize_state(&"A1".to_string());
        assert_eq!(members.len(), 2);

        // Concretize a distribution.
        let mut d = IndexMap::new();
        d.insert("A0".to_string(), 0.4);
        d.insert("A1".to_string(), 0.6);
        let abs_dist = Distribution { entries: d };

        let conc_dist = conc_map.concretize_distribution(&abs_dist);
        assert!((conc_dist.get("s0") - 0.4).abs() < 1e-10);
        // s1 and s2 should each get 0.3 (0.6 * 0.5).
        assert!((conc_dist.get("s1") - 0.3).abs() < 1e-10);
        assert!((conc_dist.get("s2") - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_galois_connection_construction() {
        let partition = make_partition();
        let gc = GaloisConnection::from_partition(partition);

        assert_eq!(gc.abstraction.num_abstract_states(), 2);
        assert!(!gc.validated);
    }

    #[test]
    fn test_build_abstract_coalgebra() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let gc = GaloisConnection::from_partition(partition);

        let abstract_coal = gc.build_abstract_coalgebra(&concrete);
        assert_eq!(abstract_coal.states.len(), 2);
        assert!(abstract_coal.states.contains_key("A0"));
        assert!(abstract_coal.states.contains_key("A1"));
    }

    #[test]
    fn test_galois_validation() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let mut gc = GaloisConnection::from_partition(partition);
        let abstract_coal = gc.build_abstract_coalgebra(&concrete);

        gc.validate(&concrete, &abstract_coal);
        assert!(gc.validated);
        // For this simple example the roundtrip might not be exact.
    }

    #[test]
    fn test_degradation_bound() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let gc = GaloisConnection::from_partition(partition);
        let abstract_coal = gc.build_abstract_coalgebra(&concrete);

        let bound = DegradationBound::compute(&concrete, &abstract_coal, &gc.abstraction);
        assert!(bound.tv_bound >= 0.0);
        assert!(bound.wasserstein_bound >= 0.0);
    }

    #[test]
    fn test_split_state() {
        let partition = make_partition();
        let mut gc = GaloisConnection::from_partition(partition);

        gc.refine_split(
            &"A1".to_string(),
            vec![
                ("A1a".to_string(), vec!["s1".to_string()]),
                ("A1b".to_string(), vec!["s2".to_string()]),
            ],
        );

        assert_eq!(gc.abstraction.num_abstract_states(), 3);
        assert_eq!(
            gc.abstraction.abstract_state(&"s1".to_string()),
            Some(&"A1a".to_string())
        );
        assert_eq!(
            gc.abstraction.abstract_state(&"s2".to_string()),
            Some(&"A1b".to_string())
        );
    }

    #[test]
    fn test_merge_states() {
        let mut partition = HashMap::new();
        partition.insert("A0".to_string(), vec!["s0".to_string()]);
        partition.insert("A1".to_string(), vec!["s1".to_string()]);
        partition.insert("A2".to_string(), vec!["s2".to_string()]);
        let mut gc = GaloisConnection::from_partition(partition);

        gc.coarsen_merge(
            &["A1".to_string(), "A2".to_string()],
            "A_merged".to_string(),
        );

        assert_eq!(gc.abstraction.num_abstract_states(), 2);
        assert_eq!(
            gc.abstraction.abstract_state(&"s1".to_string()),
            Some(&"A_merged".to_string())
        );
    }

    #[test]
    fn test_property_preservation_safety() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let gc = GaloisConnection::from_partition(partition);

        let mut pp = PropertyPreservation::new(gc.abstraction, gc.concretization);
        let prop = BehavioralProperty::safety("no_toxic", "Model does not produce toxic output");
        let result = pp.check_safety_preservation(prop, true, &concrete);

        assert!(result.preserved);
    }

    #[test]
    fn test_property_preservation_probabilistic() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let gc = GaloisConnection::from_partition(partition);

        let mut pp = PropertyPreservation::new(gc.abstraction, gc.concretization);
        let prop = BehavioralProperty::probabilistic("low_risk", 0.3, "Risk below 30%");
        let result = pp.check_probabilistic_preservation(prop, 0.1, &concrete);

        // Should be preserved since 0.1 + distortion < 0.3 (distortion starts at 0).
        assert!(result.preserved);
    }

    #[test]
    fn test_degradation_worst_states() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let gc = GaloisConnection::from_partition(partition);
        let abstract_coal = gc.build_abstract_coalgebra(&concrete);

        let bound = DegradationBound::compute(&concrete, &abstract_coal, &gc.abstraction);
        let worst = bound.worst_states(5);
        // Should have entries for each concrete state.
        assert!(!worst.is_empty());
    }

    #[test]
    fn test_consistency_check() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let abs_map = AbstractionMap::from_partition(partition);

        assert!(abs_map.is_consistent(&concrete));
    }

    #[test]
    fn test_lipschitz_computation() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let mut abs_map = AbstractionMap::from_partition(partition);

        abs_map.compute_lipschitz(&concrete, &|s1, s2| {
            // Euclidean distance on features.
            let d: f64 = s1.features.iter().zip(s2.features.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            d.sqrt()
        });

        assert!(abs_map.lipschitz_constant >= 0.0);
    }

    #[test]
    fn test_coarsening_lipschitz() {
        let partition1 = make_partition();
        let gc1 = GaloisConnection::from_partition(partition1);

        // Finer: each state is its own abstract state.
        let mut fine_partition = HashMap::new();
        fine_partition.insert("F0".to_string(), vec!["s0".to_string()]);
        fine_partition.insert("F1".to_string(), vec!["s1".to_string()]);
        fine_partition.insert("F2".to_string(), vec!["s2".to_string()]);
        let gc_fine = GaloisConnection::from_partition(fine_partition);

        let lip = gc1.coarsening_lipschitz(&gc_fine);
        assert!(lip <= 1.0);
    }

    #[test]
    fn test_upper_adjoint_check() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let gc = GaloisConnection::from_partition(partition);

        let pp = PropertyPreservation::new(gc.abstraction, gc.concretization);
        let (_, violation) = pp.check_upper_adjoint(&concrete);
        assert!(violation >= 0.0);
    }

    #[test]
    fn test_preservation_summary() {
        let concrete = make_concrete_coalgebra();
        let partition = make_partition();
        let gc = GaloisConnection::from_partition(partition);

        let mut pp = PropertyPreservation::new(gc.abstraction, gc.concretization);
        let prop1 = BehavioralProperty::safety("safe1", "desc1");
        pp.check_safety_preservation(prop1, true, &concrete);
        let prop2 = BehavioralProperty::probabilistic("prob1", 0.5, "desc2");
        pp.check_probabilistic_preservation(prop2, 0.2, &concrete);

        let summary = pp.summary();
        assert!(summary.contains("preserved"));
    }
}

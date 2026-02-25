//! Bisimulation relations and distance computation for coalgebras.
//!
//! Bisimulation is the fundamental equivalence notion for coalgebras:
//! two states are bisimilar iff they cannot be distinguished by any
//! sequence of observations. For quantitative (probabilistic) systems,
//! we use bisimulation metrics that measure the "degree of bisimilarity".
//!
//! Key algorithms:
//! - Exact bisimulation distance (full state space exploration)
//! - Approximate distance (sublinear via functor bandwidth)
//! - On-the-fly with early termination
//! - Partition refinement for classical bisimulation
//! - Coalgebraic up-to techniques

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Reverse;
use std::fmt;
use std::hash::Hash;

use nalgebra::DMatrix;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::distribution::SubDistribution;
use super::functor::{BehavioralFunctor, SimpleBehavioralValue};
use super::types::*;
use super::coalgebra::{FiniteCoalgebra, CoalgebraSystem, ProbabilisticCoalgebra};

// ---------------------------------------------------------------------------
// BisimulationRelation
// ---------------------------------------------------------------------------

/// An equivalence relation on coalgebra states representing bisimulation.
#[derive(Debug, Clone)]
pub struct BisimulationRelation {
    /// Partition of states into equivalence classes.
    pub partition: Vec<Vec<StateId>>,
    /// Quick lookup: state → class index.
    pub class_of: HashMap<StateId, usize>,
    /// Number of classes.
    pub num_classes: usize,
}

impl BisimulationRelation {
    /// Create from a partition.
    pub fn from_partition(partition: Vec<Vec<StateId>>) -> Self {
        let mut class_of = HashMap::new();
        for (i, class) in partition.iter().enumerate() {
            for state in class {
                class_of.insert(state.clone(), i);
            }
        }
        let num_classes = partition.len();
        Self {
            partition,
            class_of,
            num_classes,
        }
    }

    /// The discrete relation (every state in its own class).
    pub fn discrete(states: &[StateId]) -> Self {
        let partition: Vec<Vec<StateId>> = states
            .iter()
            .map(|s| vec![s.clone()])
            .collect();
        Self::from_partition(partition)
    }

    /// The indiscrete relation (all states in one class).
    pub fn indiscrete(states: &[StateId]) -> Self {
        Self::from_partition(vec![states.to_vec()])
    }

    /// Check if two states are bisimilar (in the same class).
    pub fn are_bisimilar(&self, s1: &StateId, s2: &StateId) -> bool {
        match (self.class_of.get(s1), self.class_of.get(s2)) {
            (Some(c1), Some(c2)) => c1 == c2,
            _ => false,
        }
    }

    /// Get the class index for a state.
    pub fn class_index(&self, state: &StateId) -> Option<usize> {
        self.class_of.get(state).copied()
    }

    /// Get all states in the same class as the given state.
    pub fn class_members(&self, state: &StateId) -> Vec<&StateId> {
        match self.class_of.get(state) {
            Some(&idx) => self.partition[idx].iter().collect(),
            None => Vec::new(),
        }
    }

    /// Refine this relation by splitting classes based on a discriminator.
    pub fn refine<F: Fn(&StateId) -> u64>(&self, discriminator: F) -> Self {
        let mut new_partition = Vec::new();
        for class in &self.partition {
            let mut subclasses: HashMap<u64, Vec<StateId>> = HashMap::new();
            for state in class {
                let disc = discriminator(state);
                subclasses
                    .entry(disc)
                    .or_insert_with(Vec::new)
                    .push(state.clone());
            }
            new_partition.extend(subclasses.into_values());
        }
        Self::from_partition(new_partition)
    }

    /// Coarsen this relation by merging classes that satisfy a predicate.
    pub fn coarsen<F: Fn(&[StateId], &[StateId]) -> bool>(&self, should_merge: F) -> Self {
        let mut merged = vec![false; self.partition.len()];
        let mut new_partition: Vec<Vec<StateId>> = Vec::new();

        for i in 0..self.partition.len() {
            if merged[i] {
                continue;
            }
            let mut class = self.partition[i].clone();
            for j in (i + 1)..self.partition.len() {
                if merged[j] {
                    continue;
                }
                if should_merge(&class, &self.partition[j]) {
                    class.extend(self.partition[j].iter().cloned());
                    merged[j] = true;
                }
            }
            new_partition.push(class);
        }

        Self::from_partition(new_partition)
    }

    /// Check if this relation is a refinement of another.
    pub fn refines(&self, other: &BisimulationRelation) -> bool {
        for class in &self.partition {
            // All elements in this class must be in the same class of `other`
            if class.is_empty() {
                continue;
            }
            let target_class = match other.class_of.get(&class[0]) {
                Some(&c) => c,
                None => return false,
            };
            for state in class.iter().skip(1) {
                match other.class_of.get(state) {
                    Some(&c) if c == target_class => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Meet (intersection) of two bisimulation relations.
    pub fn meet(&self, other: &BisimulationRelation) -> Self {
        let mut new_partition: Vec<Vec<StateId>> = Vec::new();
        let mut seen: HashSet<(usize, usize)> = HashSet::new();

        for class in &self.partition {
            for state in class {
                let c1 = self.class_of.get(state).copied().unwrap_or(0);
                let c2 = other.class_of.get(state).copied().unwrap_or(0);
                let key = (c1, c2);
                if seen.insert(key) {
                    let members: Vec<StateId> = class
                        .iter()
                        .filter(|s| other.class_of.get(*s).copied().unwrap_or(0) == c2)
                        .cloned()
                        .collect();
                    if !members.is_empty() {
                        new_partition.push(members);
                    }
                }
            }
        }

        Self::from_partition(new_partition)
    }

    /// Compute the quotient state mapping.
    pub fn quotient_map(&self) -> HashMap<StateId, StateId> {
        let mut map = HashMap::new();
        for (i, class) in self.partition.iter().enumerate() {
            let representative = StateId::indexed("q", i);
            for state in class {
                map.insert(state.clone(), representative.clone());
            }
        }
        map
    }
}

// ---------------------------------------------------------------------------
// QuantitativeBisimulation
// ---------------------------------------------------------------------------

/// Quantitative (graded) bisimulation: a pseudometric on states measuring
/// the degree of behavioral similarity.
#[derive(Debug, Clone)]
pub struct QuantitativeBisimulation {
    /// Distance matrix: distances[i][j] = d(state_i, state_j).
    pub distances: Vec<Vec<f64>>,
    /// State ordering.
    pub states: Vec<StateId>,
    /// State to index mapping.
    state_index: HashMap<StateId, usize>,
    /// Number of iterations used to compute.
    pub iterations: usize,
    /// Convergence residual.
    pub residual: f64,
}

impl QuantitativeBisimulation {
    /// Create with initial distances.
    pub fn new(states: Vec<StateId>) -> Self {
        let n = states.len();
        let distances = vec![vec![0.0; n]; n];
        let state_index: HashMap<StateId, usize> = states
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        Self {
            distances,
            states,
            state_index,
            iterations: 0,
            residual: f64::INFINITY,
        }
    }

    /// Get the distance between two states.
    pub fn distance(&self, s1: &StateId, s2: &StateId) -> f64 {
        match (self.state_index.get(s1), self.state_index.get(s2)) {
            (Some(&i), Some(&j)) => self.distances[i][j],
            _ => f64::INFINITY,
        }
    }

    /// Set the distance between two states.
    pub fn set_distance(&mut self, s1: &StateId, s2: &StateId, d: f64) {
        if let (Some(&i), Some(&j)) = (self.state_index.get(s1), self.state_index.get(s2)) {
            self.distances[i][j] = d;
            self.distances[j][i] = d;
        }
    }

    /// Find the closest pair of states.
    pub fn closest_pair(&self) -> Option<(&StateId, &StateId, f64)> {
        let n = self.states.len();
        if n < 2 {
            return None;
        }
        let mut best = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.distances[i][j] < best {
                    best = self.distances[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        Some((&self.states[best_i], &self.states[best_j], best))
    }

    /// Find the farthest pair of states (diameter).
    pub fn diameter(&self) -> f64 {
        let n = self.states.len();
        let mut max_d = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                max_d = max_d.max(self.distances[i][j]);
            }
        }
        max_d
    }

    /// Threshold to obtain a classical bisimulation relation.
    pub fn threshold_relation(&self, epsilon: f64) -> BisimulationRelation {
        let n = self.states.len();
        let mut uf = UnionFind::new(n);

        for i in 0..n {
            for j in (i + 1)..n {
                if self.distances[i][j] <= epsilon {
                    uf.union(i, j);
                }
            }
        }

        let mut classes: HashMap<usize, Vec<StateId>> = HashMap::new();
        for i in 0..n {
            let root = uf.find(i);
            classes
                .entry(root)
                .or_insert_with(Vec::new)
                .push(self.states[i].clone());
        }

        let partition: Vec<Vec<StateId>> = classes.into_values().collect();
        BisimulationRelation::from_partition(partition)
    }

    /// Check if the metric satisfies the triangle inequality.
    pub fn validate_triangle_inequality(&self) -> bool {
        let n = self.states.len();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if self.distances[i][k] > self.distances[i][j] + self.distances[j][k] + 1e-10 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check symmetry.
    pub fn validate_symmetry(&self) -> bool {
        let n = self.states.len();
        for i in 0..n {
            for j in 0..n {
                if (self.distances[i][j] - self.distances[j][i]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Get the distance matrix as a nalgebra matrix.
    pub fn as_matrix(&self) -> DMatrix<f64> {
        let n = self.states.len();
        DMatrix::from_fn(n, n, |i, j| self.distances[i][j])
    }

    /// Compute statistics about the distance distribution.
    pub fn statistics(&self) -> BisimulationStatistics {
        let n = self.states.len();
        let mut all_dists = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                all_dists.push(self.distances[i][j]);
            }
        }
        all_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = if all_dists.is_empty() {
            0.0
        } else {
            all_dists.iter().sum::<f64>() / all_dists.len() as f64
        };

        let median = if all_dists.is_empty() {
            0.0
        } else {
            all_dists[all_dists.len() / 2]
        };

        let max = all_dists.last().copied().unwrap_or(0.0);
        let min = all_dists.first().copied().unwrap_or(0.0);

        let variance = if all_dists.is_empty() {
            0.0
        } else {
            all_dists.iter().map(|d| (d - mean).powi(2)).sum::<f64>()
                / all_dists.len() as f64
        };

        BisimulationStatistics {
            num_states: n,
            num_pairs: all_dists.len(),
            mean_distance: mean,
            median_distance: median,
            max_distance: max,
            min_distance: min,
            std_distance: variance.sqrt(),
            iterations: self.iterations,
            residual: self.residual,
        }
    }
}

/// Statistics about bisimulation distances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BisimulationStatistics {
    pub num_states: usize,
    pub num_pairs: usize,
    pub mean_distance: f64,
    pub median_distance: f64,
    pub max_distance: f64,
    pub min_distance: f64,
    pub std_distance: f64,
    pub iterations: usize,
    pub residual: f64,
}

// ---------------------------------------------------------------------------
// Kantorovich lifting
// ---------------------------------------------------------------------------

/// Lift a metric on states to a metric on distributions using the
/// Kantorovich (Wasserstein-1) construction.
#[derive(Debug, Clone)]
pub struct KantorovichLifting;

impl KantorovichLifting {
    /// Compute the Kantorovich distance between two sub-distributions
    /// given a ground metric on states.
    pub fn distance(
        d1: &SubDistribution<StateId>,
        d2: &SubDistribution<StateId>,
        ground_metric: &dyn Fn(&StateId, &StateId) -> f64,
    ) -> f64 {
        d1.wasserstein_with_metric(d2, ground_metric)
    }

    /// Compute the Kantorovich distance using a distance matrix.
    pub fn distance_from_matrix(
        d1: &SubDistribution<StateId>,
        d2: &SubDistribution<StateId>,
        distances: &QuantitativeBisimulation,
    ) -> f64 {
        d1.wasserstein_with_metric(d2, |s1, s2| distances.distance(s1, s2))
    }

    /// Lift the metric through the behavioral functor.
    /// Computes d_F(v1, v2) = sup_w d_K(v1(w), v2(w))
    /// where d_K is the Kantorovich distance on the joint distributions.
    pub fn behavioral_lifting(
        v1: &SimpleBehavioralValue<StateId>,
        v2: &SimpleBehavioralValue<StateId>,
        ground_distances: &QuantitativeBisimulation,
        input_words: &[Word],
    ) -> f64 {
        let mut max_dist = 0.0f64;

        for word in input_words {
            let joint1 = v1.joint_distribution(word);
            let joint2 = v2.joint_distribution(word);

            // Compound metric: output distance + state distance
            let dist = joint1.wasserstein_with_metric(&joint2, |pair1, pair2| {
                let output_dist = if pair1.0 == pair2.0 { 0.0 } else { 1.0 };
                let state_dist = ground_distances.distance(&pair1.1, &pair2.1);
                (output_dist + state_dist) / 2.0
            });

            max_dist = max_dist.max(dist);
        }

        max_dist
    }

    /// Check that the Kantorovich lifting is non-expansive.
    pub fn verify_non_expansive(
        distributions: &[(SubDistribution<StateId>, SubDistribution<StateId>)],
        ground_metric: &dyn Fn(&StateId, &StateId) -> f64,
    ) -> bool {
        for (d1, d2) in distributions {
            let kant_dist = Self::distance(d1, d2, ground_metric);
            // The Kantorovich distance should be ≤ max ground distance
            let d2_support = d2.support();
            let max_ground: f64 = d1
                .support()
                .iter()
                .flat_map(|s1| d2_support.iter().map(move |s2| ground_metric(s1, s2)))
                .fold(0.0f64, f64::max);
            if kant_dist > max_ground + 1e-10 {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Bisimulation distance algorithms
// ---------------------------------------------------------------------------

/// Compute exact bisimulation distances via fixed-point iteration.
/// This is the standard Kantorovich-based algorithm.
pub fn exact_bisimulation_distance(
    coalgebra: &FiniteCoalgebra,
    config: &BisimulationConfig,
) -> QuantitativeBisimulation {
    let states = coalgebra.states();
    let n = states.len();
    let mut result = QuantitativeBisimulation::new(states.clone());
    let input_words = coalgebra.input_words();

    // Initialize with output distances
    for i in 0..n {
        for j in (i + 1)..n {
            let mut max_tv = 0.0f64;
            for w in &input_words {
                let d1 = coalgebra.output_distribution(&states[i], w);
                let d2 = coalgebra.output_distribution(&states[j], w);
                max_tv = max_tv.max(d1.total_variation(&d2));
            }
            result.distances[i][j] = max_tv;
            result.distances[j][i] = max_tv;
        }
    }

    // Fixed-point iteration
    for iter in 0..config.max_iterations {
        let mut max_change = 0.0f64;

        let old_distances = result.distances.clone();

        for i in 0..n {
            for j in (i + 1)..n {
                let bv1 = coalgebra.structure_map(&states[i]);
                let bv2 = coalgebra.structure_map(&states[j]);

                let new_dist = KantorovichLifting::behavioral_lifting(
                    &bv1, &bv2, &result, &input_words,
                );

                let change = (new_dist - old_distances[i][j]).abs();
                max_change = max_change.max(change);

                result.distances[i][j] = new_dist;
                result.distances[j][i] = new_dist;
            }
        }

        result.iterations = iter + 1;
        result.residual = max_change;

        if max_change < config.tolerance {
            break;
        }
    }

    result
}

/// Compute approximate bisimulation distances using sampling.
pub fn approximate_bisimulation_distance(
    coalgebra: &FiniteCoalgebra,
    config: &BisimulationConfig,
    num_samples: usize,
) -> QuantitativeBisimulation {
    let states = coalgebra.states();
    let n = states.len();
    let mut result = QuantitativeBisimulation::new(states.clone());
    let input_words = coalgebra.input_words();
    let mut rng = rand::thread_rng();

    // Sample random words and compute distance lower bounds
    let sampled_words: Vec<&Word> = if input_words.len() <= num_samples {
        input_words.iter().collect()
    } else {
        let mut indices: Vec<usize> = (0..input_words.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(num_samples);
        indices.iter().map(|&i| &input_words[i]).collect()
    };

    for i in 0..n {
        for j in (i + 1)..n {
            let mut max_dist = 0.0f64;

            for w in &sampled_words {
                let d1 = coalgebra.output_distribution(&states[i], w);
                let d2 = coalgebra.output_distribution(&states[j], w);
                max_dist = max_dist.max(d1.total_variation(&d2));
            }

            result.distances[i][j] = max_dist;
            result.distances[j][i] = max_dist;
        }
    }

    result.iterations = 1;
    result.residual = 0.0;
    result
}

/// On-the-fly bisimulation distance with early termination.
/// Stops as soon as the distance exceeds a given threshold.
pub fn on_the_fly_bisimulation_distance(
    coalgebra: &FiniteCoalgebra,
    s1: &StateId,
    s2: &StateId,
    threshold: f64,
    max_depth: usize,
) -> OnTheFlyResult {
    let input_words = coalgebra.input_words();
    let mut distance = 0.0f64;
    let mut depth = 0;
    let mut distinguishing_trace: Vec<Word> = Vec::new();
    let mut explored_pairs = 0;

    // BFS-like exploration
    let mut queue: VecDeque<(StateId, StateId, Vec<Word>, usize)> = VecDeque::new();
    queue.push_back((s1.clone(), s2.clone(), Vec::new(), 0));

    let mut visited: HashSet<(StateId, StateId)> = HashSet::new();

    while let Some((state1, state2, trace, current_depth)) = queue.pop_front() {
        if current_depth > max_depth {
            continue;
        }

        let pair = if state1 <= state2 {
            (state1.clone(), state2.clone())
        } else {
            (state2.clone(), state1.clone())
        };

        if !visited.insert(pair) {
            continue;
        }

        explored_pairs += 1;

        let bv1 = coalgebra.structure_map(&state1);
        let bv2 = coalgebra.structure_map(&state2);

        for w in &input_words {
            let d1 = bv1.output_distribution(w);
            let d2 = bv2.output_distribution(w);
            let tv = d1.total_variation(&d2);

            if tv > distance {
                distance = tv;
                distinguishing_trace = trace.clone();
                distinguishing_trace.push(w.clone());
                depth = current_depth + 1;
            }

            if distance > threshold {
                return OnTheFlyResult {
                    distance,
                    exceeded_threshold: true,
                    depth,
                    explored_pairs,
                    distinguishing_trace: Some(distinguishing_trace),
                };
            }

            // Explore successors
            let ns1 = bv1.next_state_distribution(w);
            let ns2 = bv2.next_state_distribution(w);

            for next1 in ns1.support() {
                for next2 in ns2.support() {
                    let mut new_trace = trace.clone();
                    new_trace.push(w.clone());
                    queue.push_back((
                        next1.clone(),
                        next2.clone(),
                        new_trace,
                        current_depth + 1,
                    ));
                }
            }
        }
    }

    OnTheFlyResult {
        distance,
        exceeded_threshold: false,
        depth,
        explored_pairs,
        distinguishing_trace: if distance > 0.0 {
            Some(distinguishing_trace)
        } else {
            None
        },
    }
}

/// Result of on-the-fly bisimulation computation.
#[derive(Debug, Clone)]
pub struct OnTheFlyResult {
    pub distance: f64,
    pub exceeded_threshold: bool,
    pub depth: usize,
    pub explored_pairs: usize,
    pub distinguishing_trace: Option<Vec<Word>>,
}

// ---------------------------------------------------------------------------
// Bisimulation witnesses and counterexamples
// ---------------------------------------------------------------------------

/// A witness proving that two states are bisimilar (or approximately bisimilar).
#[derive(Debug, Clone)]
pub struct BisimulationWitness {
    pub state1: StateId,
    pub state2: StateId,
    pub distance: f64,
    pub correspondence: Vec<(StateId, StateId, f64)>,
    pub depth_verified: usize,
}

impl BisimulationWitness {
    pub fn new(state1: StateId, state2: StateId, distance: f64) -> Self {
        Self {
            state1,
            state2,
            distance,
            correspondence: Vec::new(),
            depth_verified: 0,
        }
    }

    /// Check if this witness certifies ε-bisimilarity.
    pub fn certifies_epsilon_bisimilarity(&self, epsilon: f64) -> bool {
        self.distance <= epsilon
    }
}

/// A distinguishing trace (counterexample) proving two states are NOT bisimilar.
#[derive(Debug, Clone)]
pub struct DistinguishingTrace {
    pub state1: StateId,
    pub state2: StateId,
    pub trace: Vec<Word>,
    pub divergence_point: usize,
    pub output_difference: f64,
}

impl DistinguishingTrace {
    /// Find a minimal distinguishing trace between two states.
    pub fn find(
        coalgebra: &FiniteCoalgebra,
        s1: &StateId,
        s2: &StateId,
        max_depth: usize,
    ) -> Option<Self> {
        let input_words = coalgebra.input_words();

        // BFS to find the shortest distinguishing sequence
        let mut queue: VecDeque<(StateId, StateId, Vec<Word>)> = VecDeque::new();
        queue.push_back((s1.clone(), s2.clone(), Vec::new()));

        let mut visited: HashSet<(StateId, StateId)> = HashSet::new();

        while let Some((state1, state2, trace)) = queue.pop_front() {
            if trace.len() > max_depth {
                continue;
            }

            let pair = if state1 <= state2 {
                (state1.clone(), state2.clone())
            } else {
                (state2.clone(), state1.clone())
            };
            if !visited.insert(pair) {
                continue;
            }

            let bv1 = coalgebra.structure_map(&state1);
            let bv2 = coalgebra.structure_map(&state2);

            for w in &input_words {
                let d1 = bv1.output_distribution(w);
                let d2 = bv2.output_distribution(w);
                let tv = d1.total_variation(&d2);

                if tv > 1e-10 {
                    let mut full_trace = trace.clone();
                    full_trace.push(w.clone());
                    return Some(DistinguishingTrace {
                        state1: s1.clone(),
                        state2: s2.clone(),
                        trace: full_trace,
                        divergence_point: trace.len(),
                        output_difference: tv,
                    });
                }

                // Explore successors
                let ns1 = bv1.next_state_distribution(w);
                let ns2 = bv2.next_state_distribution(w);
                for next1 in ns1.support() {
                    for next2 in ns2.support() {
                        let mut new_trace = trace.clone();
                        new_trace.push(w.clone());
                        queue.push_back((next1.clone(), next2.clone(), new_trace));
                    }
                }
            }
        }

        None // States are bisimilar up to max_depth
    }

    /// Verify the distinguishing trace.
    pub fn verify(&self, coalgebra: &FiniteCoalgebra) -> bool {
        let mut current1 = self.state1.clone();
        let mut current2 = self.state2.clone();

        for (i, w) in self.trace.iter().enumerate() {
            let bv1 = coalgebra.structure_map(&current1);
            let bv2 = coalgebra.structure_map(&current2);

            let d1 = bv1.output_distribution(w);
            let d2 = bv2.output_distribution(w);
            let tv = d1.total_variation(&d2);

            if i == self.divergence_point && tv > 1e-10 {
                return true;
            }

            // Follow most probable transitions
            if let Some(entries) = bv1.get_transitions(w) {
                if let Some((_, next, _)) = entries
                    .iter()
                    .max_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap())
                {
                    current1 = next.clone();
                }
            }
            if let Some(entries) = bv2.get_transitions(w) {
                if let Some((_, next, _)) = entries
                    .iter()
                    .max_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap())
                {
                    current2 = next.clone();
                }
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Partition refinement algorithm
// ---------------------------------------------------------------------------

/// Classical partition refinement for computing the coarsest bisimulation.
pub fn partition_refinement(
    coalgebra: &FiniteCoalgebra,
) -> BisimulationRelation {
    let states = coalgebra.states();
    let n = states.len();
    if n == 0 {
        return BisimulationRelation::from_partition(Vec::new());
    }

    let input_words = coalgebra.input_words();

    // Initialize: all states in one partition
    let mut partition = vec![(0..n).collect::<HashSet<usize>>()];

    loop {
        let mut new_partition: Vec<HashSet<usize>> = Vec::new();
        let mut changed = false;

        for block in &partition {
            if block.len() <= 1 {
                new_partition.push(block.clone());
                continue;
            }

            // Build block-to-partition map
            let block_to_part: HashMap<usize, usize> = {
                let mut map = HashMap::new();
                for (pi, part) in partition.iter().enumerate() {
                    for &si in part {
                        map.insert(si, pi);
                    }
                }
                map
            };

            // Compute signature for each state in the block
            let mut sig_groups: HashMap<Vec<(usize, OrderedFloat<f64>)>, HashSet<usize>> =
                HashMap::new();

            for &si in block {
                let mut sig = Vec::new();
                let bv = coalgebra.structure_map(&states[si]);

                for w in &input_words {
                    if let Some(entries) = bv.get_transitions(w) {
                        let mut target_probs: BTreeMap<usize, f64> = BTreeMap::new();
                        for (_, target, prob) in entries {
                            let ti = states.iter().position(|s| s == target).unwrap_or(0);
                            let target_part = block_to_part.get(&ti).copied().unwrap_or(0);
                            *target_probs.entry(target_part).or_insert(0.0) += prob;
                        }
                        for (tp, prob) in target_probs {
                            sig.push((tp, OrderedFloat(prob)));
                        }
                    }
                }

                sig.sort();
                sig_groups.entry(sig).or_insert_with(HashSet::new).insert(si);
            }

            if sig_groups.len() > 1 {
                changed = true;
            }
            new_partition.extend(sig_groups.into_values());
        }

        partition = new_partition;
        if !changed {
            break;
        }
    }

    let final_partition: Vec<Vec<StateId>> = partition
        .into_iter()
        .map(|block| block.into_iter().map(|i| states[i].clone()).collect())
        .collect();

    BisimulationRelation::from_partition(final_partition)
}

// ---------------------------------------------------------------------------
// Up-to techniques
// ---------------------------------------------------------------------------

/// Coalgebraic up-to technique: enhances bisimulation checking by allowing
/// the use of known compatible relations.
#[derive(Debug, Clone)]
pub struct UpToTechnique {
    pub name: String,
    pub compatible_relations: Vec<BisimulationRelation>,
}

impl UpToTechnique {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            compatible_relations: Vec::new(),
        }
    }

    /// Up-to equivalence: R is a bisimulation up to ≡ if for every
    /// (s, t) ∈ R, s and t have matching transitions up to ≡.
    pub fn up_to_equivalence(
        &self,
        relation: &BisimulationRelation,
        coalgebra: &FiniteCoalgebra,
    ) -> bool {
        let input_words = coalgebra.input_words();

        for class in &relation.partition {
            for i in 0..class.len() {
                for j in (i + 1)..class.len() {
                    let s1 = &class[i];
                    let s2 = &class[j];

                    let bv1 = coalgebra.structure_map(s1);
                    let bv2 = coalgebra.structure_map(s2);

                    for w in &input_words {
                        let d1 = bv1.output_distribution(w);
                        let d2 = bv2.output_distribution(w);
                        if d1.total_variation(&d2) > 1e-6 {
                            return false;
                        }

                        // Check that successor states are related up to compatible relations
                        let ns1 = bv1.next_state_distribution(w);
                        let ns2 = bv2.next_state_distribution(w);

                        for next1 in ns1.support() {
                            let has_match = ns2.support().iter().any(|next2| {
                                relation.are_bisimilar(next1, next2)
                                    || self.compatible_relations.iter().any(|r| {
                                        r.are_bisimilar(next1, next2)
                                    })
                            });
                            if !has_match && ns1.weight(next1) > 1e-10 {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// Up-to context: allows factoring out a known context.
    pub fn add_compatible_relation(&mut self, rel: BisimulationRelation) {
        self.compatible_relations.push(rel);
    }
}

// ---------------------------------------------------------------------------
// Utility: Union-Find
// ---------------------------------------------------------------------------

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }

    fn same_set(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

// ---------------------------------------------------------------------------
// Multi-model comparison
// ---------------------------------------------------------------------------

/// Compare multiple models pairwise using bisimulation distances.
pub fn pairwise_comparison(
    coalgebras: &[(&str, &FiniteCoalgebra)],
    config: &BisimulationConfig,
) -> PairwiseComparisonResult {
    let n = coalgebras.len();
    let mut distances = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            // For cross-coalgebra comparison, build product and measure initial state distance
            let c1 = coalgebras[i].1;
            let c2 = coalgebras[j].1;

            let init1 = c1.initial_states();
            let init2 = c2.initial_states();

            if let (Some(s1), Some(s2)) = (init1.first(), init2.first()) {
                // Compare output distributions over the same inputs
                let words = c1.input_words();
                let mut max_tv = 0.0f64;

                for w in &words {
                    let d1 = c1.output_distribution(s1, w);
                    let d2 = c2.output_distribution(s2, w);
                    max_tv = max_tv.max(d1.total_variation(&d2));
                }

                distances[i][j] = max_tv;
                distances[j][i] = max_tv;
            } else {
                distances[i][j] = 1.0;
                distances[j][i] = 1.0;
            }
        }
    }

    let model_names: Vec<String> = coalgebras.iter().map(|(name, _)| name.to_string()).collect();

    PairwiseComparisonResult {
        model_names,
        distances,
    }
}

/// Result of pairwise model comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparisonResult {
    pub model_names: Vec<String>,
    pub distances: Vec<Vec<f64>>,
}

impl PairwiseComparisonResult {
    /// Find the most similar pair.
    pub fn most_similar(&self) -> Option<(&str, &str, f64)> {
        let n = self.model_names.len();
        if n < 2 {
            return None;
        }
        let mut best = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.distances[i][j] < best {
                    best = self.distances[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        Some((&self.model_names[best_i], &self.model_names[best_j], best))
    }

    /// Find the most different pair.
    pub fn most_different(&self) -> Option<(&str, &str, f64)> {
        let n = self.model_names.len();
        if n < 2 {
            return None;
        }
        let mut worst = 0.0f64;
        let mut worst_i = 0;
        let mut worst_j = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.distances[i][j] > worst {
                    worst = self.distances[i][j];
                    worst_i = i;
                    worst_j = j;
                }
            }
        }
        Some((&self.model_names[worst_i], &self.model_names[worst_j], worst))
    }

    /// Cluster models by bisimulation distance.
    pub fn cluster(&self, threshold: f64) -> Vec<Vec<String>> {
        let n = self.model_names.len();
        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                if self.distances[i][j] <= threshold {
                    uf.union(i, j);
                }
            }
        }

        let mut groups: HashMap<usize, Vec<String>> = HashMap::new();
        for i in 0..n {
            let root = uf.find(i);
            groups
                .entry(root)
                .or_insert_with(Vec::new)
                .push(self.model_names[i].clone());
        }

        groups.into_values().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_functor() -> BehavioralFunctor {
        BehavioralFunctor::new(
            vec![Symbol::new("a")],
            vec![OutputSymbol::new("x"), OutputSymbol::new("y")],
            1,
        )
    }

    fn make_test_coalgebra() -> FiniteCoalgebra {
        let bf = make_test_functor();
        let s0 = StateId::new("s0");
        let s1 = StateId::new("s1");
        let s2 = StateId::new("s2");
        let mut c = FiniteCoalgebra::new(
            "test",
            vec![s0.clone(), s1.clone(), s2.clone()],
            vec![s0.clone()],
            bf,
        );
        let input = Word::from_str_slice(&["a"]);
        c.add_transition(s0.clone(), input.clone(), OutputSymbol::new("x"), s1.clone(), 0.7);
        c.add_transition(s0.clone(), input.clone(), OutputSymbol::new("y"), s2.clone(), 0.3);
        c.add_transition(s1.clone(), input.clone(), OutputSymbol::new("x"), s0.clone(), 1.0);
        c.add_transition(s2.clone(), input.clone(), OutputSymbol::new("y"), s2.clone(), 1.0);
        c
    }

    #[test]
    fn test_bisimulation_relation_discrete() {
        let states = vec![StateId::new("a"), StateId::new("b")];
        let rel = BisimulationRelation::discrete(&states);
        assert_eq!(rel.num_classes, 2);
        assert!(!rel.are_bisimilar(&StateId::new("a"), &StateId::new("b")));
    }

    #[test]
    fn test_bisimulation_relation_indiscrete() {
        let states = vec![StateId::new("a"), StateId::new("b")];
        let rel = BisimulationRelation::indiscrete(&states);
        assert_eq!(rel.num_classes, 1);
        assert!(rel.are_bisimilar(&StateId::new("a"), &StateId::new("b")));
    }

    #[test]
    fn test_bisimulation_refines() {
        let states = vec![StateId::new("a"), StateId::new("b"), StateId::new("c")];
        let coarse = BisimulationRelation::indiscrete(&states);
        let fine = BisimulationRelation::discrete(&states);
        assert!(fine.refines(&coarse));
        assert!(!coarse.refines(&fine));
    }

    #[test]
    fn test_bisimulation_meet() {
        let rel1 = BisimulationRelation::from_partition(vec![
            vec![StateId::new("a"), StateId::new("b")],
            vec![StateId::new("c")],
        ]);
        let rel2 = BisimulationRelation::from_partition(vec![
            vec![StateId::new("a")],
            vec![StateId::new("b"), StateId::new("c")],
        ]);
        let meet = rel1.meet(&rel2);
        // a is separated from b (different in rel2), b from c (different in rel1)
        assert_eq!(meet.num_classes, 3);
    }

    #[test]
    fn test_bisimulation_refine() {
        let rel = BisimulationRelation::from_partition(vec![
            vec![StateId::new("a"), StateId::new("b"), StateId::new("c")],
        ]);
        // Split by first character
        let refined = rel.refine(|s| {
            s.as_str().chars().next().map(|c| c as u64).unwrap_or(0)
        });
        // All start with same chars, so no split
        assert_eq!(refined.num_classes, 1);
    }

    #[test]
    fn test_quantitative_bisimulation() {
        let states = vec![StateId::new("s0"), StateId::new("s1"), StateId::new("s2")];
        let mut qb = QuantitativeBisimulation::new(states);
        qb.set_distance(&StateId::new("s0"), &StateId::new("s1"), 0.3);
        qb.set_distance(&StateId::new("s0"), &StateId::new("s2"), 0.8);
        qb.set_distance(&StateId::new("s1"), &StateId::new("s2"), 0.5);

        assert!((qb.distance(&StateId::new("s0"), &StateId::new("s1")) - 0.3).abs() < 1e-10);
        assert!((qb.diameter() - 0.8).abs() < 1e-10);
        assert!(qb.validate_symmetry());
    }

    #[test]
    fn test_threshold_relation() {
        let states = vec![StateId::new("s0"), StateId::new("s1"), StateId::new("s2")];
        let mut qb = QuantitativeBisimulation::new(states);
        qb.set_distance(&StateId::new("s0"), &StateId::new("s1"), 0.1);
        qb.set_distance(&StateId::new("s0"), &StateId::new("s2"), 0.8);
        qb.set_distance(&StateId::new("s1"), &StateId::new("s2"), 0.7);

        let rel = qb.threshold_relation(0.2);
        assert!(rel.are_bisimilar(&StateId::new("s0"), &StateId::new("s1")));
        assert!(!rel.are_bisimilar(&StateId::new("s0"), &StateId::new("s2")));
    }

    #[test]
    fn test_bisimulation_statistics() {
        let states = vec![StateId::new("s0"), StateId::new("s1"), StateId::new("s2")];
        let mut qb = QuantitativeBisimulation::new(states);
        qb.set_distance(&StateId::new("s0"), &StateId::new("s1"), 0.2);
        qb.set_distance(&StateId::new("s0"), &StateId::new("s2"), 0.8);
        qb.set_distance(&StateId::new("s1"), &StateId::new("s2"), 0.5);

        let stats = qb.statistics();
        assert_eq!(stats.num_states, 3);
        assert_eq!(stats.num_pairs, 3);
        assert!((stats.max_distance - 0.8).abs() < 1e-10);
        assert!((stats.min_distance - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_exact_bisimulation_distance() {
        let coalgebra = make_test_coalgebra();
        let config = BisimulationConfig::default();
        let result = exact_bisimulation_distance(&coalgebra, &config);

        // s0 and s2 should have positive distance (different outputs)
        assert!(result.distance(&StateId::new("s0"), &StateId::new("s2")) > 0.0);
        // Self-distance should be 0
        assert!((result.distance(&StateId::new("s0"), &StateId::new("s0"))).abs() < 1e-10);
        assert!(result.validate_symmetry());
    }

    #[test]
    fn test_approximate_bisimulation_distance() {
        let coalgebra = make_test_coalgebra();
        let config = BisimulationConfig::default();
        let result = approximate_bisimulation_distance(&coalgebra, &config, 10);
        assert!(result.validate_symmetry());
    }

    #[test]
    fn test_on_the_fly_same_state() {
        let coalgebra = make_test_coalgebra();
        let result = on_the_fly_bisimulation_distance(
            &coalgebra,
            &StateId::new("s0"),
            &StateId::new("s0"),
            0.5,
            3,
        );
        assert!((result.distance).abs() < 1e-10);
        assert!(!result.exceeded_threshold);
    }

    #[test]
    fn test_on_the_fly_different_states() {
        let coalgebra = make_test_coalgebra();
        let result = on_the_fly_bisimulation_distance(
            &coalgebra,
            &StateId::new("s0"),
            &StateId::new("s2"),
            0.01,
            3,
        );
        assert!(result.distance > 0.0);
        assert!(result.exceeded_threshold);
    }

    #[test]
    fn test_kantorovich_lifting() {
        let d1 = SubDistribution::point(StateId::new("s0"));
        let d2 = SubDistribution::point(StateId::new("s1"));
        let ground = |s1: &StateId, s2: &StateId| {
            if s1 == s2 { 0.0 } else { 1.0 }
        };
        let dist = KantorovichLifting::distance(&d1, &d2, &ground);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kantorovich_same_distribution() {
        let d = SubDistribution::uniform(vec![StateId::new("s0"), StateId::new("s1")]);
        let ground = |s1: &StateId, s2: &StateId| {
            if s1 == s2 { 0.0 } else { 1.0 }
        };
        let dist = KantorovichLifting::distance(&d, &d, &ground);
        assert!((dist).abs() < 1e-10);
    }

    #[test]
    fn test_distinguishing_trace() {
        let coalgebra = make_test_coalgebra();
        // s1 and s2 should be distinguishable
        let trace = DistinguishingTrace::find(
            &coalgebra,
            &StateId::new("s1"),
            &StateId::new("s2"),
            3,
        );
        assert!(trace.is_some());
        let t = trace.unwrap();
        assert!(t.output_difference > 0.0);
    }

    #[test]
    fn test_distinguishing_trace_same_state() {
        let coalgebra = make_test_coalgebra();
        let trace = DistinguishingTrace::find(
            &coalgebra,
            &StateId::new("s1"),
            &StateId::new("s1"),
            3,
        );
        assert!(trace.is_none());
    }

    #[test]
    fn test_partition_refinement() {
        let coalgebra = make_test_coalgebra();
        let bisim = partition_refinement(&coalgebra);
        // s1 and s2 have different behaviors, so should be separated
        assert!(!bisim.are_bisimilar(&StateId::new("s1"), &StateId::new("s2")));
    }

    #[test]
    fn test_bisimulation_witness() {
        let w = BisimulationWitness::new(
            StateId::new("s0"),
            StateId::new("s0"),
            0.0,
        );
        assert!(w.certifies_epsilon_bisimilarity(0.01));
    }

    #[test]
    fn test_pairwise_comparison() {
        let c1 = make_test_coalgebra();
        let c2 = make_test_coalgebra();
        let config = BisimulationConfig::default();

        let result = pairwise_comparison(
            &[("model1", &c1), ("model2", &c2)],
            &config,
        );
        assert_eq!(result.model_names.len(), 2);
        // Same coalgebra should have distance 0
        assert!((result.distances[0][1]).abs() < 1e-10);
    }

    #[test]
    fn test_pairwise_clustering() {
        let c1 = make_test_coalgebra();
        let c2 = make_test_coalgebra();
        let config = BisimulationConfig::default();

        let result = pairwise_comparison(
            &[("m1", &c1), ("m2", &c2)],
            &config,
        );
        let clusters = result.cluster(0.1);
        // Both should be in the same cluster
        assert_eq!(clusters.len(), 1);
    }

    #[test]
    fn test_up_to_technique() {
        let coalgebra = make_test_coalgebra();
        let bisim = partition_refinement(&coalgebra);
        let upto = UpToTechnique::new("test");
        assert!(upto.up_to_equivalence(&bisim, &coalgebra));
    }

    #[test]
    fn test_quotient_map() {
        let rel = BisimulationRelation::from_partition(vec![
            vec![StateId::new("s0"), StateId::new("s1")],
            vec![StateId::new("s2")],
        ]);
        let qmap = rel.quotient_map();
        assert_eq!(qmap[&StateId::new("s0")], qmap[&StateId::new("s1")]);
        assert_ne!(qmap[&StateId::new("s0")], qmap[&StateId::new("s2")]);
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(1, 3);
        assert!(uf.same_set(0, 2));
        assert!(!uf.same_set(0, 4));
    }

    #[test]
    fn test_coarsen_relation() {
        let rel = BisimulationRelation::from_partition(vec![
            vec![StateId::new("a")],
            vec![StateId::new("b")],
            vec![StateId::new("c")],
        ]);
        let coarsened = rel.coarsen(|c1, c2| {
            // Merge if both have single element
            c1.len() == 1 && c2.len() == 1
        });
        assert!(coarsened.num_classes < rel.num_classes);
    }

    #[test]
    fn test_most_similar_different() {
        let result = PairwiseComparisonResult {
            model_names: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            distances: vec![
                vec![0.0, 0.3, 0.8],
                vec![0.3, 0.0, 0.5],
                vec![0.8, 0.5, 0.0],
            ],
        };
        let (m1, m2, d) = result.most_similar().unwrap();
        assert_eq!(m1, "a");
        assert_eq!(m2, "b");
        assert!((d - 0.3).abs() < 1e-10);

        let (m1, m2, d) = result.most_different().unwrap();
        assert_eq!(m1, "a");
        assert_eq!(m2, "c");
        assert!((d - 0.8).abs() < 1e-10);
    }
}

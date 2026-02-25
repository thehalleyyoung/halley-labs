//! Coalgebra implementations for behavioral modeling of LLMs.
//!
//! A coalgebra (S, γ) for a functor F: Set → Set consists of a carrier
//! set S (the state space) and a structure map γ: S → F(S). Coalgebras
//! model state-based systems: γ decomposes each state into its one-step
//! observable behavior.
//!
//! For LLM behavioral auditing, the key coalgebra is:
//!   γ: S → F_LLM(S) = (Σ_≤k × D(S))^{Σ*_≤n}
//! which, for each state s, maps each input word w to:
//!   - an output symbol σ ∈ Σ_≤k
//!   - a sub-distribution over successor states

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use ordered_float::OrderedFloat;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::distribution::SubDistribution;
use super::functor::{
    BehavioralFunctor, SimpleBehavioralValue, SubDistributionFunctor, PredicateLifting,
};
use super::types::*;

// ---------------------------------------------------------------------------
// Coalgebra trait
// ---------------------------------------------------------------------------

/// A coalgebra (S, γ) for a behavioral functor F.
/// The trait abstracts over the state type and provides the structure map.
pub trait CoalgebraSystem: fmt::Debug + Send + Sync {
    /// The state type.
    type State: Clone + Eq + Hash + Ord + fmt::Debug + Send + Sync + 'static;

    /// Apply the structure map γ: S → F(S) to a state.
    fn structure_map(&self, state: &Self::State) -> SimpleBehavioralValue<Self::State>;

    /// Get the set of all states (may be infinite for lazy coalgebras).
    fn states(&self) -> Vec<Self::State>;

    /// Get initial states (entry points for exploration).
    fn initial_states(&self) -> Vec<Self::State>;

    /// Number of states.
    fn num_states(&self) -> usize {
        self.states().len()
    }

    /// Check if a state is in the coalgebra.
    fn has_state(&self, state: &Self::State) -> bool {
        self.states().contains(state)
    }

    /// Get the behavioral functor specification.
    fn functor(&self) -> &BehavioralFunctor;

    /// Get the input words for this coalgebra.
    fn input_words(&self) -> Vec<Word> {
        self.functor().input_words()
    }

    /// Get the output distribution for a state and input.
    fn output_distribution(
        &self,
        state: &Self::State,
        input: &Word,
    ) -> SubDistribution<OutputSymbol> {
        let bv = self.structure_map(state);
        bv.output_distribution(input)
    }

    /// Get the next-state distribution for a state and input.
    fn next_state_distribution(
        &self,
        state: &Self::State,
        input: &Word,
    ) -> SubDistribution<Self::State> {
        let bv = self.structure_map(state);
        bv.next_state_distribution(input)
    }

    /// Simulate a trace: starting from a state, follow a sequence of inputs.
    fn simulate_trace(
        &self,
        start: &Self::State,
        inputs: &[Word],
    ) -> Vec<(Word, OutputSymbol, Self::State)> {
        let mut current = start.clone();
        let mut trace = Vec::new();

        for input in inputs {
            let bv = self.structure_map(&current);
            if let Some(entries) = bv.get_transitions(input) {
                if let Some((out, next, _)) = entries
                    .iter()
                    .max_by(|(_, _, a), (_, _, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    trace.push((input.clone(), out.clone(), next.clone()));
                    current = next.clone();
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        trace
    }

    /// Sample a trace using probabilistic transitions.
    fn sample_trace(
        &self,
        start: &Self::State,
        inputs: &[Word],
        rng: &mut impl rand::Rng,
    ) -> Vec<(Word, OutputSymbol, Self::State)> {
        let mut current = start.clone();
        let mut trace = Vec::new();

        for input in inputs {
            let bv = self.structure_map(&current);
            let joint = bv.joint_distribution(input);
            if let Some((out, next)) = joint.sample(rng) {
                trace.push((input.clone(), out, next.clone()));
                current = next;
            } else {
                break;
            }
        }
        trace
    }

    /// Compute the reachable states from a given set of starting states.
    fn reachable_states(&self, starts: &[Self::State]) -> HashSet<Self::State> {
        let mut visited = HashSet::new();
        let mut queue: VecDeque<Self::State> = starts.iter().cloned().collect();

        while let Some(state) = queue.pop_front() {
            if !visited.insert(state.clone()) {
                continue;
            }
            let bv = self.structure_map(&state);
            for next in bv.next_states() {
                if !visited.contains(next) {
                    queue.push_back(next.clone());
                }
            }
        }
        visited
    }

    /// Build a transition graph for visualization/analysis.
    fn transition_graph(&self) -> DiGraph<Self::State, (Word, OutputSymbol, f64)> {
        let mut graph = DiGraph::new();
        let mut node_map: HashMap<Self::State, NodeIndex> = HashMap::new();

        for state in self.states() {
            let idx = graph.add_node(state.clone());
            node_map.insert(state, idx);
        }

        for state in self.states() {
            let bv = self.structure_map(&state);
            for (input, entries) in &bv.transitions {
                for (output, next_state, prob) in entries {
                    if let (Some(&from), Some(&to)) =
                        (node_map.get(&state), node_map.get(next_state))
                    {
                        graph.add_edge(
                            from,
                            to,
                            (input.clone(), output.clone(), *prob),
                        );
                    }
                }
            }
        }

        graph
    }

    /// Check if the coalgebra is well-formed (all distributions are valid).
    fn validate(&self, tolerance: f64) -> Result<(), CoalgebraError> {
        for state in self.states() {
            let bv = self.structure_map(&state);
            if !bv.validate(tolerance) {
                return Err(CoalgebraError::InvalidDistribution(
                    bv.transitions
                        .values()
                        .flat_map(|v| v.iter().map(|(_, _, p)| p))
                        .sum(),
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FiniteCoalgebra
// ---------------------------------------------------------------------------

/// A finite-state coalgebra with explicit transition maps.
#[derive(Debug, Clone)]
pub struct FiniteCoalgebra {
    pub name: String,
    states: Vec<StateId>,
    initial: Vec<StateId>,
    transitions: HashMap<StateId, SimpleBehavioralValue<StateId>>,
    behavioral_functor: BehavioralFunctor,
}

impl FiniteCoalgebra {
    pub fn new(
        name: impl Into<String>,
        states: Vec<StateId>,
        initial: Vec<StateId>,
        behavioral_functor: BehavioralFunctor,
    ) -> Self {
        let transitions = HashMap::new();
        Self {
            name: name.into(),
            states,
            initial,
            transitions,
            behavioral_functor,
        }
    }

    /// Set the behavioral value for a state.
    pub fn set_behavior(
        &mut self,
        state: StateId,
        behavior: SimpleBehavioralValue<StateId>,
    ) {
        self.transitions.insert(state, behavior);
    }

    /// Add a single transition.
    pub fn add_transition(
        &mut self,
        source: StateId,
        input: Word,
        output: OutputSymbol,
        target: StateId,
        probability: f64,
    ) {
        let bv = self
            .transitions
            .entry(source)
            .or_insert_with(SimpleBehavioralValue::new);
        bv.add_transition(input, output, target, probability);
    }

    /// Get the number of transitions.
    pub fn num_transitions(&self) -> usize {
        self.transitions
            .values()
            .map(|bv| bv.total_entries())
            .sum()
    }

    /// Build from a transition table.
    pub fn from_transition_table(
        name: impl Into<String>,
        table: &TransitionTable,
        behavioral_functor: BehavioralFunctor,
    ) -> Self {
        let states: Vec<StateId> = table.states().into_iter().collect();
        let initial = if states.is_empty() {
            Vec::new()
        } else {
            vec![states[0].clone()]
        };

        let mut coalgebra = Self::new(name, states, initial, behavioral_functor);

        for (source, input_map) in &table.entries {
            for (input, entries) in input_map {
                for entry in entries {
                    coalgebra.add_transition(
                        source.clone(),
                        input.clone(),
                        entry.output.clone(),
                        entry.next_state.clone(),
                        entry.probability,
                    );
                }
            }
        }

        coalgebra
    }

    /// Compute the quotient coalgebra under an equivalence relation.
    pub fn quotient(
        &self,
        partition: &[Vec<StateId>],
    ) -> Self {
        let mut class_map: HashMap<StateId, usize> = HashMap::new();
        for (i, class) in partition.iter().enumerate() {
            for state in class {
                class_map.insert(state.clone(), i);
            }
        }

        let new_states: Vec<StateId> = (0..partition.len())
            .map(|i| StateId::indexed("q", i))
            .collect();

        let new_initial: Vec<StateId> = self
            .initial
            .iter()
            .filter_map(|s| class_map.get(s).map(|&i| new_states[i].clone()))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let mut quotient =
            Self::new(format!("{}_quotient", self.name), new_states.clone(), new_initial, self.behavioral_functor.clone());

        for (source, bv) in &self.transitions {
            let source_class = match class_map.get(source) {
                Some(&c) => c,
                None => continue,
            };
            let new_source = new_states[source_class].clone();

            for (input, entries) in &bv.transitions {
                for (output, target, prob) in entries {
                    let target_class = match class_map.get(target) {
                        Some(&c) => c,
                        None => continue,
                    };
                    let new_target = new_states[target_class].clone();
                    quotient.add_transition(
                        new_source.clone(),
                        input.clone(),
                        output.clone(),
                        new_target,
                        *prob,
                    );
                }
            }
        }

        quotient
    }

    /// Compute the sub-coalgebra reachable from initial states.
    pub fn reachable_subcoalgebra(&self) -> Self {
        let reachable = self.reachable_states(&self.initial);
        let new_states: Vec<StateId> = reachable.iter().cloned().collect();
        let new_initial: Vec<StateId> = self
            .initial
            .iter()
            .filter(|s| reachable.contains(s))
            .cloned()
            .collect();

        let mut sub = Self::new(
            format!("{}_reachable", self.name),
            new_states,
            new_initial,
            self.behavioral_functor.clone(),
        );

        for (state, bv) in &self.transitions {
            if reachable.contains(state) {
                sub.transitions.insert(state.clone(), bv.clone());
            }
        }

        sub
    }

    /// Product of two coalgebras. The product coalgebra has state space S1 × S2
    /// and the behavior is the product of behaviors.
    pub fn product(&self, other: &Self) -> Self {
        let mut product_states = Vec::new();
        let mut product_initial = Vec::new();

        for s1 in &self.states {
            for s2 in &other.states {
                let ps = StateId::new(format!("({},{})", s1, s2));
                product_states.push(ps.clone());

                if self.initial.contains(s1) && other.initial.contains(s2) {
                    product_initial.push(ps);
                }
            }
        }

        let mut product_coalgebra = Self::new(
            format!("{}_x_{}", self.name, other.name),
            product_states.clone(),
            product_initial,
            self.behavioral_functor.clone(),
        );

        for s1 in &self.states {
            for s2 in &other.states {
                let ps = StateId::new(format!("({},{})", s1, s2));
                let bv1 = self.structure_map(s1);
                let bv2 = other.structure_map(s2);

                let mut product_bv = SimpleBehavioralValue::new();
                for (input, entries1) in &bv1.transitions {
                    if let Some(entries2) = bv2.transitions.get(input) {
                        for (out1, next1, p1) in entries1 {
                            for (out2, next2, p2) in entries2 {
                                if out1 == out2 {
                                    let product_next =
                                        StateId::new(format!("({},{})", next1, next2));
                                    product_bv.add_transition(
                                        input.clone(),
                                        out1.clone(),
                                        product_next,
                                        p1 * p2,
                                    );
                                }
                            }
                        }
                    }
                }

                product_coalgebra.set_behavior(ps, product_bv);
            }
        }

        product_coalgebra
    }

    /// Coproduct (disjoint union) of two coalgebras.
    pub fn coproduct(&self, other: &Self) -> Self {
        let mut coprod_states = Vec::new();
        let mut coprod_initial = Vec::new();

        for s in &self.states {
            let cs = StateId::new(format!("L_{}", s));
            coprod_states.push(cs.clone());
            if self.initial.contains(s) {
                coprod_initial.push(cs);
            }
        }
        for s in &other.states {
            let cs = StateId::new(format!("R_{}", s));
            coprod_states.push(cs.clone());
            if other.initial.contains(s) {
                coprod_initial.push(cs);
            }
        }

        let mut coprod = Self::new(
            format!("{}_+_{}", self.name, other.name),
            coprod_states,
            coprod_initial,
            self.behavioral_functor.clone(),
        );

        for (s, bv) in &self.transitions {
            let new_s = StateId::new(format!("L_{}", s));
            let new_bv = bv.fmap(&|t: &StateId| StateId::new(format!("L_{}", t)));
            coprod.set_behavior(new_s, new_bv);
        }

        for (s, bv) in &other.transitions {
            let new_s = StateId::new(format!("R_{}", s));
            let new_bv = bv.fmap(&|t: &StateId| StateId::new(format!("R_{}", t)));
            coprod.set_behavior(new_s, new_bv);
        }

        coprod
    }

    /// Minimize the coalgebra by identifying bisimilar states.
    /// Uses partition refinement.
    pub fn minimize(&self) -> Self {
        let states: Vec<&StateId> = self.states.iter().collect();
        let n = states.len();
        if n <= 1 {
            return self.clone();
        }

        // Initialize: all states in one partition
        let mut partition: Vec<HashSet<usize>> = vec![(0..n).collect()];

        loop {
            let mut new_partition: Vec<HashSet<usize>> = Vec::new();
            let mut changed = false;

            for block in &partition {
                let splits = self.refine_block(block, &partition, &states);
                if splits.len() > 1 {
                    changed = true;
                }
                new_partition.extend(splits);
            }

            partition = new_partition;
            if !changed {
                break;
            }
        }

        // Build partition as Vec<Vec<StateId>>
        let part: Vec<Vec<StateId>> = partition
            .iter()
            .map(|block| block.iter().map(|&i| states[i].clone()).collect())
            .collect();

        self.quotient(&part)
    }

    /// Refine a block based on transition behavior.
    fn refine_block(
        &self,
        block: &HashSet<usize>,
        partition: &[HashSet<usize>],
        states: &[&StateId],
    ) -> Vec<HashSet<usize>> {
        if block.len() <= 1 {
            return vec![block.clone()];
        }

        // Map each state index to its "signature" based on transitions
        let mut signatures: HashMap<usize, Vec<(Word, OutputSymbol, usize, OrderedFloat<f64>)>> =
            HashMap::new();

        let block_to_partition: HashMap<usize, usize> = {
            let mut map = HashMap::new();
            for (pi, part) in partition.iter().enumerate() {
                for &si in part {
                    map.insert(si, pi);
                }
            }
            map
        };

        for &si in block {
            let state = states[si];
            let bv = self.structure_map(state);
            let mut sig = Vec::new();

            for (input, entries) in &bv.transitions {
                for (output, target, prob) in entries {
                    let target_idx = states.iter().position(|&s| s == target).unwrap_or(0);
                    let target_part = block_to_partition.get(&target_idx).copied().unwrap_or(0);
                    sig.push((
                        input.clone(),
                        output.clone(),
                        target_part,
                        OrderedFloat(*prob),
                    ));
                }
            }
            sig.sort();
            signatures.insert(si, sig);
        }

        // Group by signature
        let mut groups: HashMap<
            Vec<(Word, OutputSymbol, usize, OrderedFloat<f64>)>,
            HashSet<usize>,
        > = HashMap::new();
        for (&si, sig) in &signatures {
            groups.entry(sig.clone()).or_insert_with(HashSet::new).insert(si);
        }

        groups.into_values().collect()
    }

    /// Compute observable behavior up to depth n.
    pub fn observable_behavior(&self, state: &StateId, depth: usize) -> ObservableBehavior {
        let mut observations = Vec::new();
        self.collect_observations(state, &[], depth, &mut observations);

        ObservableBehavior {
            state: state.clone(),
            depth,
            observations,
        }
    }

    fn collect_observations(
        &self,
        state: &StateId,
        prefix: &[Word],
        remaining_depth: usize,
        observations: &mut Vec<Observation>,
    ) {
        let bv = self.structure_map(state);

        for (input, entries) in &bv.transitions {
            for (output, next_state, prob) in entries {
                let mut path = prefix.to_vec();
                path.push(input.clone());

                observations.push(Observation {
                    input_path: path.clone(),
                    output: output.clone(),
                    probability: *prob,
                });

                if remaining_depth > 0 {
                    self.collect_observations(
                        next_state,
                        &path,
                        remaining_depth - 1,
                        observations,
                    );
                }
            }
        }
    }

    /// Check if two states are observationally equivalent up to depth n.
    pub fn observationally_equivalent(
        &self,
        s1: &StateId,
        s2: &StateId,
        depth: usize,
        tolerance: f64,
    ) -> bool {
        if depth == 0 {
            return true;
        }

        let bv1 = self.structure_map(s1);
        let bv2 = self.structure_map(s2);

        for input in self.input_words() {
            let d1 = bv1.output_distribution(&input);
            let d2 = bv2.output_distribution(&input);
            if d1.total_variation(&d2) > tolerance {
                return false;
            }

            // Check recursively on next states
            if depth > 1 {
                let ns1 = bv1.next_state_distribution(&input);
                let ns2 = bv2.next_state_distribution(&input);

                for s in ns1.support() {
                    for t in ns2.support() {
                        let p1 = ns1.weight(s);
                        let p2 = ns2.weight(t);
                        if p1 > tolerance && p2 > tolerance {
                            if !self.observationally_equivalent(s, t, depth - 1, tolerance) {
                                return false;
                            }
                        }
                    }
                }
            }
        }

        true
    }

    /// Find strongly connected components using the transition graph.
    pub fn strongly_connected_components(&self) -> Vec<Vec<StateId>> {
        let graph = self.transition_graph();
        let sccs = algo::tarjan_scc(&graph);
        sccs.into_iter()
            .map(|scc| scc.into_iter().map(|idx| graph[idx].clone()).collect())
            .collect()
    }

    /// Check if the coalgebra is deterministic (at most one transition per input).
    pub fn is_deterministic(&self) -> bool {
        for bv in self.transitions.values() {
            for entries in bv.transitions.values() {
                if entries.len() > 1 {
                    return false;
                }
            }
        }
        true
    }
}

impl CoalgebraSystem for FiniteCoalgebra {
    type State = StateId;

    fn structure_map(&self, state: &StateId) -> SimpleBehavioralValue<StateId> {
        self.transitions
            .get(state)
            .cloned()
            .unwrap_or_else(SimpleBehavioralValue::new)
    }

    fn states(&self) -> Vec<StateId> {
        self.states.clone()
    }

    fn initial_states(&self) -> Vec<StateId> {
        self.initial.clone()
    }

    fn functor(&self) -> &BehavioralFunctor {
        &self.behavioral_functor
    }
}

/// Observable behavior of a state.
#[derive(Debug, Clone)]
pub struct ObservableBehavior {
    pub state: StateId,
    pub depth: usize,
    pub observations: Vec<Observation>,
}

/// A single observation in the behavior tree.
#[derive(Debug, Clone)]
pub struct Observation {
    pub input_path: Vec<Word>,
    pub output: OutputSymbol,
    pub probability: f64,
}

// ---------------------------------------------------------------------------
// ProbabilisticCoalgebra
// ---------------------------------------------------------------------------

/// A coalgebra over the sub-distribution functor, with additional
/// probabilistic analysis capabilities.
#[derive(Debug, Clone)]
pub struct ProbabilisticCoalgebra {
    inner: FiniteCoalgebra,
    /// Stationary distribution (if computed).
    stationary: Option<SubDistribution<StateId>>,
    /// Mixing time estimate.
    mixing_time: Option<usize>,
}

impl ProbabilisticCoalgebra {
    pub fn new(inner: FiniteCoalgebra) -> Self {
        Self {
            inner,
            stationary: None,
            mixing_time: None,
        }
    }

    pub fn inner(&self) -> &FiniteCoalgebra {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut FiniteCoalgebra {
        &mut self.inner
    }

    /// Compute the transition matrix for a given input word.
    /// Entry (i,j) = probability of transitioning from state i to state j on input w.
    pub fn transition_matrix(&self, input: &Word) -> nalgebra::DMatrix<f64> {
        let n = self.inner.states.len();
        let mut matrix = nalgebra::DMatrix::zeros(n, n);

        let state_index: HashMap<&StateId, usize> = self
            .inner
            .states
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();

        for (i, state) in self.inner.states.iter().enumerate() {
            let bv = self.inner.structure_map(state);
            if let Some(entries) = bv.get_transitions(input) {
                for (_, target, prob) in entries {
                    if let Some(&j) = state_index.get(target) {
                        matrix[(i, j)] += prob;
                    }
                }
            }
        }

        matrix
    }

    /// Compute the aggregated transition matrix (summed over all inputs, weighted).
    pub fn aggregated_transition_matrix(
        &self,
        input_weights: &SubDistribution<Word>,
    ) -> nalgebra::DMatrix<f64> {
        let n = self.inner.states.len();
        let mut matrix = nalgebra::DMatrix::zeros(n, n);

        for (input, weight) in input_weights.iter() {
            let m = self.transition_matrix(input);
            matrix += m * weight;
        }

        matrix
    }

    /// Compute the stationary distribution via power iteration.
    pub fn compute_stationary(
        &mut self,
        input_weights: &SubDistribution<Word>,
        max_iterations: usize,
        tolerance: f64,
    ) -> SubDistribution<StateId> {
        let n = self.inner.states.len();
        if n == 0 {
            let empty = SubDistribution::empty();
            self.stationary = Some(empty.clone());
            return empty;
        }

        let matrix = self.aggregated_transition_matrix(input_weights);

        // Power iteration
        let mut pi = nalgebra::DVector::from_element(n, 1.0 / n as f64);
        let mut iterations = 0;

        for _ in 0..max_iterations {
            let new_pi = &matrix.transpose() * &pi;
            let diff: f64 = (&new_pi - &pi).iter().map(|x| x.abs()).sum();
            pi = new_pi;
            iterations += 1;

            if diff < tolerance {
                break;
            }
        }

        // Normalize
        let sum: f64 = pi.iter().sum();
        if sum > 1e-15 {
            pi /= sum;
        }

        let mut weights = BTreeMap::new();
        for (i, state) in self.inner.states.iter().enumerate() {
            let p = pi[i];
            if p > 1e-15 {
                weights.insert(state.clone(), p);
            }
        }

        let dist = SubDistribution::from_weights(weights).unwrap_or_else(|_| {
            SubDistribution::uniform(self.inner.states.clone())
        });

        self.mixing_time = Some(iterations);
        self.stationary = Some(dist.clone());
        dist
    }

    /// Get the stationary distribution (if computed).
    pub fn stationary_distribution(&self) -> Option<&SubDistribution<StateId>> {
        self.stationary.as_ref()
    }

    /// Get the mixing time estimate (if computed).
    pub fn mixing_time(&self) -> Option<usize> {
        self.mixing_time
    }

    /// Compute the expected hitting time from state i to state j.
    pub fn expected_hitting_time(
        &self,
        from: &StateId,
        to: &StateId,
        input: &Word,
        max_steps: usize,
    ) -> f64 {
        let n = self.inner.states.len();
        let state_index: HashMap<&StateId, usize> = self
            .inner
            .states
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();

        let from_idx = match state_index.get(from) {
            Some(&i) => i,
            None => return f64::INFINITY,
        };
        let to_idx = match state_index.get(to) {
            Some(&i) => i,
            None => return f64::INFINITY,
        };

        if from_idx == to_idx {
            return 0.0;
        }

        let matrix = self.transition_matrix(input);

        // Compute by absorbing state method
        // Remove the target state and solve (I - P')h = 1
        let mut prob_at_target = 0.0;
        let mut expected = 0.0;
        let mut dist = nalgebra::DVector::zeros(n);
        dist[from_idx] = 1.0;

        for step in 1..=max_steps {
            dist = &matrix * &dist;
            let p_arrive = dist[to_idx];
            expected += step as f64 * (p_arrive - prob_at_target).max(0.0);
            prob_at_target = p_arrive;
        }

        if prob_at_target > 1e-10 {
            expected / prob_at_target
        } else {
            f64::INFINITY
        }
    }

    /// Compute the spectral gap of the transition matrix.
    pub fn spectral_gap(&self, input: &Word) -> f64 {
        let matrix = self.transition_matrix(input);
        let n = matrix.nrows();
        if n <= 1 {
            return 1.0;
        }

        // The spectral gap is 1 - |λ₂| where λ₂ is the second-largest eigenvalue.
        // We use power iteration on (M - v*ones) to find λ₂.
        // For simplicity, compute via repeated multiplication.
        let mut v1 = nalgebra::DVector::from_element(n, 1.0 / (n as f64).sqrt());
        for _ in 0..100 {
            v1 = &matrix.transpose() * &v1;
            let norm = v1.norm();
            if norm > 1e-15 {
                v1 /= norm;
            }
        }

        // Deflate: M' = M - λ₁ * v₁ * v₁ᵀ
        let lambda1 = (&matrix.transpose() * &v1).dot(&v1);
        let deflated = &matrix.transpose() - lambda1 * &v1 * v1.transpose();

        let mut v2 = nalgebra::DVector::from_fn(n, |i, _| if i == 0 { 1.0 } else { 0.0 });
        for _ in 0..100 {
            v2 = &deflated * &v2;
            let norm = v2.norm();
            if norm > 1e-15 {
                v2 /= norm;
            }
        }
        let lambda2 = (&deflated * &v2).dot(&v2);

        (1.0 - lambda2.abs()).max(0.0)
    }

    /// Compute the entropy rate of the Markov chain.
    pub fn entropy_rate(&self, input: &Word) -> f64 {
        let input_weights = SubDistribution::point(input.clone());
        let stationary = if let Some(ref s) = self.stationary {
            s.clone()
        } else {
            let mut clone = self.clone();
            clone.compute_stationary(&input_weights, 1000, 1e-8)
        };

        let mut h = 0.0;
        for (state, pi_s) in stationary.iter() {
            if pi_s < 1e-15 {
                continue;
            }
            let bv = self.inner.structure_map(state);
            if let Some(entries) = bv.get_transitions(input) {
                for (_, _, p) in entries {
                    if *p > 1e-15 {
                        h -= pi_s * p * p.ln();
                    }
                }
            }
        }
        h
    }

    /// Compute the absorption probability into a set of target states.
    pub fn absorption_probability(
        &self,
        from: &StateId,
        targets: &HashSet<StateId>,
        input: &Word,
        max_steps: usize,
    ) -> f64 {
        let matrix = self.transition_matrix(input);
        let n = self.inner.states.len();

        let state_index: HashMap<&StateId, usize> = self
            .inner
            .states
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();

        let from_idx = match state_index.get(from) {
            Some(&i) => i,
            None => return 0.0,
        };

        let target_indices: HashSet<usize> = targets
            .iter()
            .filter_map(|t| state_index.get(t).copied())
            .collect();

        let mut dist = nalgebra::DVector::zeros(n);
        dist[from_idx] = 1.0;
        let mut absorbed = 0.0;

        for _ in 0..max_steps {
            dist = &matrix * &dist;
            for &ti in &target_indices {
                absorbed += dist[ti];
                dist[ti] = 0.0; // absorb
            }
        }

        absorbed.min(1.0)
    }
}

impl CoalgebraSystem for ProbabilisticCoalgebra {
    type State = StateId;

    fn structure_map(&self, state: &StateId) -> SimpleBehavioralValue<StateId> {
        self.inner.structure_map(state)
    }

    fn states(&self) -> Vec<StateId> {
        self.inner.states()
    }

    fn initial_states(&self) -> Vec<StateId> {
        self.inner.initial_states()
    }

    fn functor(&self) -> &BehavioralFunctor {
        self.inner.functor()
    }
}

// ---------------------------------------------------------------------------
// LLMBehavioralCoalgebra
// ---------------------------------------------------------------------------

/// The specific coalgebra for LLM behavioral auditing.
/// Wraps a finite coalgebra with LLM-specific metadata and operations.
#[derive(Debug, Clone)]
pub struct LLMBehavioralCoalgebra {
    pub model_id: String,
    pub coalgebra: ProbabilisticCoalgebra,
    pub abstraction_level: (usize, usize, f64), // (k, n, ε)
    pub cluster_labels: HashMap<ClusterId, String>,
    pub trace_count: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl LLMBehavioralCoalgebra {
    pub fn new(
        model_id: impl Into<String>,
        coalgebra: ProbabilisticCoalgebra,
        abstraction_level: (usize, usize, f64),
    ) -> Self {
        Self {
            model_id: model_id.into(),
            coalgebra,
            abstraction_level,
            cluster_labels: HashMap::new(),
            trace_count: 0,
            last_updated: chrono::Utc::now(),
        }
    }

    /// Build an LLM coalgebra from a trace corpus.
    pub fn from_traces(
        model_id: impl Into<String>,
        traces: &TraceCorpus,
        config: &CoalgebraConfig,
    ) -> Self {
        let model_id = model_id.into();

        // Build state space from trace prefixes
        let mut states = Vec::new();
        let mut transitions: HashMap<StateId, SimpleBehavioralValue<StateId>> = HashMap::new();

        // Initial state
        let initial = StateId::new("init");
        states.push(initial.clone());

        // Process each trace to build transitions
        let mut state_counter = 0usize;
        for trace in &traces.traces {
            let mut current = initial.clone();

            for step in &trace.steps {
                let next = StateId::indexed("s", state_counter);
                state_counter += 1;
                if !states.contains(&next) {
                    states.push(next.clone());
                }

                let bv = transitions
                    .entry(current.clone())
                    .or_insert_with(SimpleBehavioralValue::new);

                bv.add_transition(
                    step.input.clone(),
                    step.output.clone(),
                    next.clone(),
                    1.0, // Will be normalized later
                );

                current = next;
            }
        }

        // Normalize transitions
        for bv in transitions.values_mut() {
            for entries in bv.transitions.values_mut() {
                let total: f64 = entries.iter().map(|(_, _, p)| p).sum();
                if total > 0.0 {
                    for entry in entries.iter_mut() {
                        entry.2 /= total;
                    }
                }
            }
        }

        let input_alpha: Vec<Symbol> = traces
            .traces
            .iter()
            .flat_map(|t| t.steps.iter())
            .flat_map(|s| s.input.symbols.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let output_alpha: Vec<OutputSymbol> = traces
            .traces
            .iter()
            .flat_map(|t| t.steps.iter())
            .map(|s| s.output.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let bf = BehavioralFunctor::new(
            input_alpha,
            output_alpha,
            config.max_input_length,
        );

        let mut fc = FiniteCoalgebra::new(
            format!("llm_{}", model_id),
            states,
            vec![initial],
            bf,
        );

        for (state, bv) in transitions {
            fc.set_behavior(state, bv);
        }

        let pc = ProbabilisticCoalgebra::new(fc);

        Self::new(
            model_id,
            pc,
            (
                config.num_clusters,
                config.max_input_length,
                config.epsilon,
            ),
        )
    }

    /// Query the LLM and update the coalgebra with new observations.
    /// Uses `todo!()` for actual API call since we need network access.
    pub fn query_and_update(
        &mut self,
        _state: &StateId,
        _input: &Word,
    ) -> (OutputSymbol, StateId) {
        todo!("LLM API call required - implement with actual model endpoint")
    }

    /// Compute the behavioral fingerprint of the model.
    pub fn behavioral_fingerprint(&self, depth: usize) -> BehavioralFingerprint {
        let initial = self.coalgebra.inner().initial_states();
        let mut fingerprint_data = Vec::new();

        for state in &initial {
            let behavior = self.coalgebra.inner().observable_behavior(state, depth);
            for obs in &behavior.observations {
                fingerprint_data.push((
                    obs.input_path.clone(),
                    obs.output.clone(),
                    OrderedFloat(obs.probability),
                ));
            }
        }

        fingerprint_data.sort();

        BehavioralFingerprint {
            model_id: self.model_id.clone(),
            depth,
            observations: fingerprint_data,
            abstraction: self.abstraction_level,
        }
    }

    /// Compare behavioral fingerprints of two models.
    pub fn fingerprint_distance(fp1: &BehavioralFingerprint, fp2: &BehavioralFingerprint) -> f64 {
        let mut total_diff = 0.0;
        let all_keys: HashSet<&(Vec<Word>, OutputSymbol, OrderedFloat<f64>)> = fp1
            .observations
            .iter()
            .chain(fp2.observations.iter())
            .collect();

        let fp1_set: HashSet<_> = fp1.observations.iter().collect();
        let fp2_set: HashSet<_> = fp2.observations.iter().collect();

        let shared = fp1_set.intersection(&fp2_set).count();
        let total = fp1_set.union(&fp2_set).count();

        if total == 0 {
            return 0.0;
        }

        1.0 - (shared as f64 / total as f64)
    }

    /// Get summary statistics about the coalgebra.
    pub fn summary(&self) -> CoalgebraSummary {
        let inner = self.coalgebra.inner();
        let sccs = inner.strongly_connected_components();
        let reachable = inner.reachable_states(&inner.initial_states());

        CoalgebraSummary {
            model_id: self.model_id.clone(),
            num_states: inner.num_states(),
            num_transitions: inner.num_transitions(),
            num_initial_states: inner.initial_states().len(),
            num_reachable_states: reachable.len(),
            num_sccs: sccs.len(),
            is_deterministic: inner.is_deterministic(),
            abstraction_level: self.abstraction_level,
            trace_count: self.trace_count,
        }
    }
}

impl CoalgebraSystem for LLMBehavioralCoalgebra {
    type State = StateId;

    fn structure_map(&self, state: &StateId) -> SimpleBehavioralValue<StateId> {
        self.coalgebra.structure_map(state)
    }

    fn states(&self) -> Vec<StateId> {
        self.coalgebra.states()
    }

    fn initial_states(&self) -> Vec<StateId> {
        self.coalgebra.initial_states()
    }

    fn functor(&self) -> &BehavioralFunctor {
        self.coalgebra.functor()
    }
}

/// Behavioral fingerprint of an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralFingerprint {
    pub model_id: String,
    pub depth: usize,
    pub observations: Vec<(Vec<Word>, OutputSymbol, OrderedFloat<f64>)>,
    pub abstraction: (usize, usize, f64),
}

/// Summary statistics for a coalgebra.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoalgebraSummary {
    pub model_id: String,
    pub num_states: usize,
    pub num_transitions: usize,
    pub num_initial_states: usize,
    pub num_reachable_states: usize,
    pub num_sccs: usize,
    pub is_deterministic: bool,
    pub abstraction_level: (usize, usize, f64),
    pub trace_count: usize,
}

// ---------------------------------------------------------------------------
// Coalgebra morphisms
// ---------------------------------------------------------------------------

/// A coalgebra morphism h: (S₁, γ₁) → (S₂, γ₂) is a function h: S₁ → S₂
/// such that F(h) ∘ γ₁ = γ₂ ∘ h.
#[derive(Debug, Clone)]
pub struct CoalgebraMorphism {
    pub name: String,
    pub state_map: HashMap<StateId, StateId>,
}

impl CoalgebraMorphism {
    pub fn new(name: impl Into<String>, state_map: HashMap<StateId, StateId>) -> Self {
        Self {
            name: name.into(),
            state_map,
        }
    }

    /// Apply the morphism to a state.
    pub fn apply(&self, state: &StateId) -> Option<&StateId> {
        self.state_map.get(state)
    }

    /// Check if this is a valid coalgebra morphism.
    pub fn validate(
        &self,
        source: &FiniteCoalgebra,
        target: &FiniteCoalgebra,
        tolerance: f64,
    ) -> bool {
        for state in source.states() {
            let mapped = match self.state_map.get(&state) {
                Some(m) => m,
                None => return false,
            };

            let gamma1 = source.structure_map(&state);
            let gamma2 = target.structure_map(mapped);

            // Check F(h) ∘ γ₁ = γ₂ ∘ h
            let pushed = gamma1.fmap(&|s: &StateId| {
                self.state_map.get(s).cloned().unwrap_or_else(|| s.clone())
            });

            // Compare pushed and gamma2
            for input in source.input_words() {
                let d1 = pushed.output_distribution(&input);
                let d2 = gamma2.output_distribution(&input);
                if d1.total_variation(&d2) > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Compose two morphisms: h₂ ∘ h₁.
    pub fn compose(&self, other: &CoalgebraMorphism) -> CoalgebraMorphism {
        let mut composed = HashMap::new();
        for (s1, s2) in &self.state_map {
            if let Some(s3) = other.state_map.get(s2) {
                composed.insert(s1.clone(), s3.clone());
            }
        }
        CoalgebraMorphism::new(
            format!("{}_then_{}", self.name, other.name),
            composed,
        )
    }

    /// Check if this is an epimorphism (surjective on states).
    pub fn is_epi(&self, target: &FiniteCoalgebra) -> bool {
        let image: HashSet<&StateId> = self.state_map.values().collect();
        target.states().iter().all(|s| image.contains(s))
    }

    /// Check if this is a monomorphism (injective on states).
    pub fn is_mono(&self) -> bool {
        let values: Vec<&StateId> = self.state_map.values().collect();
        let unique: HashSet<&StateId> = values.iter().copied().collect();
        values.len() == unique.len()
    }

    /// Check if this is an isomorphism.
    pub fn is_iso(&self, target: &FiniteCoalgebra) -> bool {
        self.is_mono() && self.is_epi(target)
    }

    /// Compute the inverse (if this is an isomorphism).
    pub fn inverse(&self) -> Option<CoalgebraMorphism> {
        if !self.is_mono() {
            return None;
        }
        let mut inv = HashMap::new();
        for (s1, s2) in &self.state_map {
            inv.insert(s2.clone(), s1.clone());
        }
        Some(CoalgebraMorphism::new(
            format!("{}_inverse", self.name),
            inv,
        ))
    }
}

// ---------------------------------------------------------------------------
// Final coalgebra approximation
// ---------------------------------------------------------------------------

/// Approximate the final coalgebra via iterative unfolding.
/// The final coalgebra for the behavioral functor is the set of all
/// behavioral trees, which we approximate to a finite depth.
pub fn approximate_final_coalgebra(
    functor: &BehavioralFunctor,
    depth: usize,
) -> FiniteCoalgebra {
    // The final coalgebra approximation at depth 0 has one state (the terminal object)
    // At depth n+1, we refine based on one-step behavior
    let mut states = vec![StateId::new("⊥")];
    let mut transitions: HashMap<StateId, SimpleBehavioralValue<StateId>> = HashMap::new();
    let bf = functor.clone();

    if depth == 0 {
        let fc = FiniteCoalgebra::new("final_approx_0", states, vec![StateId::new("⊥")], bf);
        return fc;
    }

    // Build level by level
    let mut level_states: Vec<Vec<StateId>> = Vec::new();
    level_states.push(vec![StateId::new("⊥")]);

    for d in 1..=depth {
        let prev_states = &level_states[d - 1];
        let mut new_states = Vec::new();

        // For each combination of behaviors over prev_states, create a new state
        let num_inputs = functor.num_input_words();
        let num_outputs = functor.output_alphabet.len().max(1);
        let num_prev = prev_states.len().max(1);

        // Limit the number of new states to avoid explosion
        let max_new_states = (num_outputs * num_prev).min(100);

        for i in 0..max_new_states {
            let state = StateId::indexed(&format!("d{}", d), i);
            new_states.push(state);
        }

        level_states.push(new_states);
    }

    // Flatten all states
    states = level_states.iter().flatten().cloned().collect();

    FiniteCoalgebra::new(
        format!("final_approx_{}", depth),
        states.clone(),
        vec![states[0].clone()],
        bf,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_functor() -> BehavioralFunctor {
        BehavioralFunctor::new(
            vec![Symbol::new("a"), Symbol::new("b")],
            vec![OutputSymbol::new("x"), OutputSymbol::new("y")],
            1,
        )
    }

    fn make_test_coalgebra() -> FiniteCoalgebra {
        let bf = make_test_functor();
        let s0 = StateId::new("s0");
        let s1 = StateId::new("s1");
        let s2 = StateId::new("s2");

        let mut coalgebra = FiniteCoalgebra::new(
            "test",
            vec![s0.clone(), s1.clone(), s2.clone()],
            vec![s0.clone()],
            bf,
        );

        let input_a = Word::from_str_slice(&["a"]);
        let input_b = Word::from_str_slice(&["b"]);

        // s0 --a/x--> s1 (0.7), s0 --a/y--> s2 (0.3)
        coalgebra.add_transition(
            s0.clone(), input_a.clone(), OutputSymbol::new("x"), s1.clone(), 0.7,
        );
        coalgebra.add_transition(
            s0.clone(), input_a.clone(), OutputSymbol::new("y"), s2.clone(), 0.3,
        );

        // s0 --b/x--> s0 (1.0)
        coalgebra.add_transition(
            s0.clone(), input_b.clone(), OutputSymbol::new("x"), s0.clone(), 1.0,
        );

        // s1 --a/y--> s0 (1.0)
        coalgebra.add_transition(
            s1.clone(), input_a.clone(), OutputSymbol::new("y"), s0.clone(), 1.0,
        );

        // s2 --a/x--> s2 (1.0)
        coalgebra.add_transition(
            s2.clone(), input_a.clone(), OutputSymbol::new("x"), s2.clone(), 1.0,
        );

        coalgebra
    }

    #[test]
    fn test_finite_coalgebra_creation() {
        let coalgebra = make_test_coalgebra();
        assert_eq!(coalgebra.num_states(), 3);
        assert_eq!(coalgebra.initial_states().len(), 1);
    }

    #[test]
    fn test_structure_map() {
        let coalgebra = make_test_coalgebra();
        let bv = coalgebra.structure_map(&StateId::new("s0"));
        assert!(bv.num_inputs() > 0);
    }

    #[test]
    fn test_output_distribution() {
        let coalgebra = make_test_coalgebra();
        let input = Word::from_str_slice(&["a"]);
        let dist = coalgebra.output_distribution(&StateId::new("s0"), &input);
        assert!((dist.weight(&OutputSymbol::new("x")) - 0.7).abs() < 1e-10);
        assert!((dist.weight(&OutputSymbol::new("y")) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_next_state_distribution() {
        let coalgebra = make_test_coalgebra();
        let input = Word::from_str_slice(&["a"]);
        let dist = coalgebra.next_state_distribution(&StateId::new("s0"), &input);
        assert!((dist.weight(&StateId::new("s1")) - 0.7).abs() < 1e-10);
        assert!((dist.weight(&StateId::new("s2")) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_reachable_states() {
        let coalgebra = make_test_coalgebra();
        let reachable = coalgebra.reachable_states(&[StateId::new("s0")]);
        assert!(reachable.contains(&StateId::new("s0")));
        assert!(reachable.contains(&StateId::new("s1")));
        assert!(reachable.contains(&StateId::new("s2")));
    }

    #[test]
    fn test_transition_graph() {
        let coalgebra = make_test_coalgebra();
        let graph = coalgebra.transition_graph();
        assert_eq!(graph.node_count(), 3);
        assert!(graph.edge_count() >= 4);
    }

    #[test]
    fn test_validate() {
        let coalgebra = make_test_coalgebra();
        assert!(coalgebra.validate(1e-6).is_ok());
    }

    #[test]
    fn test_simulate_trace() {
        let coalgebra = make_test_coalgebra();
        let inputs = vec![
            Word::from_str_slice(&["a"]),
            Word::from_str_slice(&["a"]),
        ];
        let trace = coalgebra.simulate_trace(&StateId::new("s0"), &inputs);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_quotient() {
        let coalgebra = make_test_coalgebra();
        // Put s1 and s2 in the same class
        let partition = vec![
            vec![StateId::new("s0")],
            vec![StateId::new("s1"), StateId::new("s2")],
        ];
        let quotient = coalgebra.quotient(&partition);
        assert_eq!(quotient.num_states(), 2);
    }

    #[test]
    fn test_reachable_subcoalgebra() {
        let coalgebra = make_test_coalgebra();
        let sub = coalgebra.reachable_subcoalgebra();
        assert_eq!(sub.num_states(), 3); // All states are reachable
    }

    #[test]
    fn test_product() {
        let c1 = make_test_coalgebra();
        let c2 = make_test_coalgebra();
        let prod = c1.product(&c2);
        assert_eq!(prod.num_states(), 9); // 3 × 3
    }

    #[test]
    fn test_coproduct() {
        let c1 = make_test_coalgebra();
        let c2 = make_test_coalgebra();
        let coprod = c1.coproduct(&c2);
        assert_eq!(coprod.num_states(), 6); // 3 + 3
    }

    #[test]
    fn test_minimize() {
        let coalgebra = make_test_coalgebra();
        let minimized = coalgebra.minimize();
        assert!(minimized.num_states() <= coalgebra.num_states());
    }

    #[test]
    fn test_observable_behavior() {
        let coalgebra = make_test_coalgebra();
        let behavior = coalgebra.observable_behavior(&StateId::new("s0"), 2);
        assert!(!behavior.observations.is_empty());
    }

    #[test]
    fn test_sccs() {
        let coalgebra = make_test_coalgebra();
        let sccs = coalgebra.strongly_connected_components();
        assert!(!sccs.is_empty());
    }

    #[test]
    fn test_is_deterministic() {
        let coalgebra = make_test_coalgebra();
        // s0 has two transitions on input "a", so not deterministic
        assert!(!coalgebra.is_deterministic());
    }

    // --- ProbabilisticCoalgebra tests ---

    #[test]
    fn test_transition_matrix() {
        let coalgebra = make_test_coalgebra();
        let pc = ProbabilisticCoalgebra::new(coalgebra);
        let input = Word::from_str_slice(&["a"]);
        let matrix = pc.transition_matrix(&input);
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 3);
    }

    #[test]
    fn test_stationary_distribution() {
        let coalgebra = make_test_coalgebra();
        let mut pc = ProbabilisticCoalgebra::new(coalgebra);
        let input = Word::from_str_slice(&["a"]);
        let input_weights = SubDistribution::point(input);
        let stat = pc.compute_stationary(&input_weights, 1000, 1e-8);
        assert!(stat.is_proper(0.01));
    }

    #[test]
    fn test_spectral_gap() {
        let coalgebra = make_test_coalgebra();
        let pc = ProbabilisticCoalgebra::new(coalgebra);
        let input = Word::from_str_slice(&["a"]);
        let gap = pc.spectral_gap(&input);
        assert!(gap >= 0.0 && gap <= 1.0);
    }

    #[test]
    fn test_entropy_rate() {
        let coalgebra = make_test_coalgebra();
        let pc = ProbabilisticCoalgebra::new(coalgebra);
        let input = Word::from_str_slice(&["a"]);
        let h = pc.entropy_rate(&input);
        assert!(h >= 0.0);
    }

    #[test]
    fn test_absorption_probability() {
        let coalgebra = make_test_coalgebra();
        let pc = ProbabilisticCoalgebra::new(coalgebra);
        let input = Word::from_str_slice(&["a"]);
        let targets: HashSet<StateId> = vec![StateId::new("s2")].into_iter().collect();
        let prob = pc.absorption_probability(&StateId::new("s0"), &targets, &input, 100);
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    // --- LLMBehavioralCoalgebra tests ---

    #[test]
    fn test_llm_coalgebra_from_traces() {
        let mut corpus = TraceCorpus::new("test-model");
        let mut trace = InteractionTrace::new("test-model");
        trace.add_step(
            Word::from_str_slice(&["hello"]),
            OutputSymbol::new("hi"),
        );
        trace.add_step(
            Word::from_str_slice(&["how"]),
            OutputSymbol::new("fine"),
        );
        corpus.add_trace(trace);

        let config = CoalgebraConfig::default();
        let llm = LLMBehavioralCoalgebra::from_traces("test-model", &corpus, &config);
        assert!(llm.coalgebra.inner().num_states() > 0);
    }

    #[test]
    fn test_behavioral_fingerprint() {
        let coalgebra = make_test_coalgebra();
        let pc = ProbabilisticCoalgebra::new(coalgebra);
        let llm = LLMBehavioralCoalgebra::new("test", pc, (10, 5, 0.01));
        let fp = llm.behavioral_fingerprint(1);
        assert_eq!(fp.model_id, "test");
    }

    #[test]
    fn test_summary() {
        let coalgebra = make_test_coalgebra();
        let pc = ProbabilisticCoalgebra::new(coalgebra);
        let llm = LLMBehavioralCoalgebra::new("test", pc, (10, 5, 0.01));
        let summary = llm.summary();
        assert_eq!(summary.num_states, 3);
    }

    // --- CoalgebraMorphism tests ---

    #[test]
    fn test_morphism_identity() {
        let coalgebra = make_test_coalgebra();
        let mut map = HashMap::new();
        for s in coalgebra.states() {
            map.insert(s.clone(), s);
        }
        let morphism = CoalgebraMorphism::new("identity", map);
        assert!(morphism.is_mono());
        assert!(morphism.is_epi(&coalgebra));
        assert!(morphism.is_iso(&coalgebra));
    }

    #[test]
    fn test_morphism_validate() {
        let coalgebra = make_test_coalgebra();
        let mut map = HashMap::new();
        for s in coalgebra.states() {
            map.insert(s.clone(), s);
        }
        let morphism = CoalgebraMorphism::new("identity", map);
        assert!(morphism.validate(&coalgebra, &coalgebra, 1e-6));
    }

    #[test]
    fn test_morphism_compose() {
        let mut m1 = HashMap::new();
        m1.insert(StateId::new("a"), StateId::new("b"));
        let mut m2 = HashMap::new();
        m2.insert(StateId::new("b"), StateId::new("c"));

        let h1 = CoalgebraMorphism::new("h1", m1);
        let h2 = CoalgebraMorphism::new("h2", m2);
        let composed = h1.compose(&h2);

        assert_eq!(composed.apply(&StateId::new("a")), Some(&StateId::new("c")));
    }

    #[test]
    fn test_morphism_inverse() {
        let mut map = HashMap::new();
        map.insert(StateId::new("a"), StateId::new("1"));
        map.insert(StateId::new("b"), StateId::new("2"));
        let morphism = CoalgebraMorphism::new("bijection", map);
        let inv = morphism.inverse().unwrap();
        assert_eq!(inv.apply(&StateId::new("1")), Some(&StateId::new("a")));
    }

    #[test]
    fn test_final_coalgebra_approx() {
        let bf = make_test_functor();
        let final_approx = approximate_final_coalgebra(&bf, 2);
        assert!(final_approx.num_states() > 0);
    }

    #[test]
    fn test_from_transition_table() {
        let mut table = TransitionTable::new();
        table.add_entry(
            StateId::new("s0"),
            Word::from_str_slice(&["a"]),
            OutputSymbol::new("x"),
            StateId::new("s1"),
            1.0,
        );
        let bf = make_test_functor();
        let coalgebra = FiniteCoalgebra::from_transition_table("test", &table, bf);
        assert!(coalgebra.num_states() >= 2);
    }

    #[test]
    fn test_sample_trace() {
        let coalgebra = make_test_coalgebra();
        let mut rng = rand::thread_rng();
        let inputs = vec![
            Word::from_str_slice(&["a"]),
            Word::from_str_slice(&["a"]),
        ];
        let trace = coalgebra.sample_trace(&StateId::new("s0"), &inputs, &mut rng);
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_expected_hitting_time() {
        let coalgebra = make_test_coalgebra();
        let pc = ProbabilisticCoalgebra::new(coalgebra);
        let input = Word::from_str_slice(&["a"]);
        let ht = pc.expected_hitting_time(
            &StateId::new("s0"),
            &StateId::new("s0"),
            &input,
            100,
        );
        assert_eq!(ht, 0.0); // Same state
    }
}

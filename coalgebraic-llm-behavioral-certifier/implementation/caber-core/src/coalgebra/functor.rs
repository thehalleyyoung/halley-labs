//! Functor implementations for coalgebraic behavioral modeling.
//!
//! In category theory, a functor F: Set → Set maps sets to sets and functions
//! to functions, preserving identity and composition. The key functors for
//! LLM behavioral modeling are:
//!
//! - SubDistributionFunctor: D(X) = {μ: X → [0,1] | Σ μ(x) ≤ 1}
//! - BehavioralFunctor: F_LLM(X) = (Σ_≤k × D(X))^{Σ*_≤n}
//!   maps each input word to an output symbol and a distribution over next states
//!
//! The behavioral functor captures the notion that an LLM, given a prompt
//! (word of bounded length), produces a response (output symbol from a
//! bounded alphabet) together with a probabilistic transition to a next state.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::distribution::SubDistribution;
use super::types::*;

// ---------------------------------------------------------------------------
// Functor trait
// ---------------------------------------------------------------------------

/// A functor F: Set → Set in the category of sets and functions.
/// Implemented concretely for finite sets.
pub trait Functor: fmt::Debug + Send + Sync {
    /// The type of elements in the codomain F(X).
    type Applied<X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static>: Clone + fmt::Debug + Send + Sync;

    /// Apply the functor to a morphism (function) f: X → Y,
    /// yielding F(f): F(X) → F(Y). This is the "map" or "pushforward".
    fn fmap<X, Y, F>(
        fx: &Self::Applied<X>,
        f: &F,
    ) -> Self::Applied<Y>
    where
        X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        Y: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        F: Fn(&X) -> Y;

    /// Name of this functor for debugging/display.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// SubDistributionFunctor
// ---------------------------------------------------------------------------

/// The sub-distribution functor D: Set → Set.
/// D(X) is the set of finitely-supported sub-distributions on X.
/// D(f)(μ)(y) = Σ_{x: f(x)=y} μ(x) (pushforward measure).
#[derive(Debug, Clone)]
pub struct SubDistributionFunctor;

impl Functor for SubDistributionFunctor {
    type Applied<X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static> = SubDistribution<X>;

    fn fmap<X, Y, F>(
        fx: &SubDistribution<X>,
        f: &F,
    ) -> SubDistribution<Y>
    where
        X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        Y: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        F: Fn(&X) -> Y,
    {
        fx.map(f)
    }

    fn name(&self) -> &str {
        "D (Sub-Distribution)"
    }
}

impl SubDistributionFunctor {
    pub fn new() -> Self {
        Self
    }

    /// Construct the unit (Dirac delta) natural transformation η: Id → D.
    pub fn unit<X: Clone + Eq + Hash + Ord>(x: X) -> SubDistribution<X> {
        SubDistribution::point(x)
    }

    /// Construct the multiplication (join) natural transformation μ: DD → D.
    /// Flattens a distribution over distributions into a single distribution.
    pub fn join<X: Clone + Eq + Hash + Ord>(
        ddx: &SubDistribution<SubDistribution<X>>,
    ) -> SubDistribution<X> {
        ddx.flat_map(|inner| inner.clone())
    }

    /// Kleisli composition: given f: X → D(Y) and g: Y → D(Z),
    /// compute g ∘_K f: X → D(Z).
    pub fn kleisli_compose<X, Y, Z, F, G>(f: F, g: G) -> impl Fn(&X) -> SubDistribution<Z>
    where
        X: Clone + Eq + Hash + Ord + 'static,
        Y: Clone + Eq + Hash + Ord + 'static,
        Z: Clone + Eq + Hash + Ord + 'static,
        F: Fn(&X) -> SubDistribution<Y> + 'static,
        G: Fn(&Y) -> SubDistribution<Z> + 'static,
    {
        move |x: &X| {
            let dy = f(x);
            dy.flat_map(|y| g(y))
        }
    }

    /// The strength natural transformation: X × D(Y) → D(X × Y).
    pub fn strength<X, Y>(
        x: &X,
        dy: &SubDistribution<Y>,
    ) -> SubDistribution<(X, Y)>
    where
        X: Clone + Eq + Hash + Ord,
        Y: Clone + Eq + Hash + Ord,
    {
        dy.map(|y| (x.clone(), y.clone()))
    }

    /// Commutative strength: D(X) × Y → D(X × Y).
    pub fn costrength<X, Y>(
        dx: &SubDistribution<X>,
        y: &Y,
    ) -> SubDistribution<(X, Y)>
    where
        X: Clone + Eq + Hash + Ord,
        Y: Clone + Eq + Hash + Ord,
    {
        dx.map(|x| (x.clone(), y.clone()))
    }

    /// Double strength: D(X) × D(Y) → D(X × Y) (independent product).
    pub fn double_strength<X, Y>(
        dx: &SubDistribution<X>,
        dy: &SubDistribution<Y>,
    ) -> SubDistribution<(X, Y)>
    where
        X: Clone + Eq + Hash + Ord,
        Y: Clone + Eq + Hash + Ord,
    {
        dx.product(dy)
    }
}

impl Default for SubDistributionFunctor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BehavioralFunctorValue: F_LLM(X) = (Σ_≤k × D(X))^{Σ*_≤n}
// ---------------------------------------------------------------------------

/// A single behavioral step: an output symbol paired with a distribution
/// over next states.
#[derive(Debug, Clone)]
pub struct BehavioralStep<X: Clone + Eq + Hash + Ord> {
    pub output: OutputSymbol,
    pub continuation: SubDistribution<X>,
}

impl<X: Clone + Eq + Hash + Ord> PartialEq for BehavioralStep<X> {
    fn eq(&self, other: &Self) -> bool {
        self.output == other.output && self.continuation == other.continuation
    }
}

impl<X: Clone + Eq + Hash + Ord> Eq for BehavioralStep<X> {}

impl<X: Clone + Eq + Hash + Ord> std::hash::Hash for BehavioralStep<X> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.output.hash(state);
        self.continuation.hash(state);
    }
}

impl<X: Clone + Eq + Hash + Ord> PartialOrd for BehavioralStep<X> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<X: Clone + Eq + Hash + Ord> Ord for BehavioralStep<X> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.output.cmp(&other.output)
            .then_with(|| self.continuation.cmp(&other.continuation))
    }
}

impl<X: Clone + Eq + Hash + Ord> BehavioralStep<X> {
    pub fn new(output: OutputSymbol, continuation: SubDistribution<X>) -> Self {
        Self {
            output,
            continuation,
        }
    }

    pub fn terminal(output: OutputSymbol) -> Self {
        Self {
            output,
            continuation: SubDistribution::empty(),
        }
    }

    pub fn map_states<Y, F>(&self, f: &F) -> BehavioralStep<Y>
    where
        Y: Clone + Eq + Hash + Ord,
        F: Fn(&X) -> Y,
    {
        BehavioralStep {
            output: self.output.clone(),
            continuation: self.continuation.map(f),
        }
    }
}

/// The value type for the behavioral functor: F_LLM(X).
/// For each input word w ∈ Σ*_≤n, we get a sub-distribution over
/// behavioral steps (output × next-state distribution).
#[derive(Debug, Clone)]
pub struct BehavioralFunctorValue<X: Clone + Eq + Hash + Ord> {
    /// Maps input words to distributions over behavioral steps.
    transitions: BTreeMap<Word, SubDistribution<BehavioralStep<X>>>,
}

// We need Eq/Hash/Ord for SubDistribution<BehavioralStep<X>>, so we implement
// a wrapper for use inside SubDistribution. Instead, we use a simpler representation.

/// Simplified behavioral value: for each input word, a distribution over (output, next_state).
#[derive(Debug, Clone)]
pub struct SimpleBehavioralValue<X: Clone + Eq + Hash + Ord> {
    pub transitions: BTreeMap<Word, Vec<(OutputSymbol, X, f64)>>,
}

impl<X: Clone + Eq + Hash + Ord + fmt::Debug> SimpleBehavioralValue<X> {
    pub fn new() -> Self {
        Self {
            transitions: BTreeMap::new(),
        }
    }

    pub fn add_transition(
        &mut self,
        input: Word,
        output: OutputSymbol,
        next_state: X,
        probability: f64,
    ) {
        let entries = self.transitions.entry(input).or_insert_with(Vec::new);
        entries.push((output, next_state, probability));
    }

    pub fn get_transitions(&self, input: &Word) -> Option<&[(OutputSymbol, X, f64)]> {
        self.transitions.get(input).map(|v| v.as_slice())
    }

    pub fn inputs(&self) -> Vec<&Word> {
        self.transitions.keys().collect()
    }

    pub fn num_inputs(&self) -> usize {
        self.transitions.len()
    }

    /// Get the output distribution for a given input (marginalizing over next states).
    pub fn output_distribution(&self, input: &Word) -> SubDistribution<OutputSymbol>
    where
        OutputSymbol: Ord,
    {
        match self.transitions.get(input) {
            None => SubDistribution::empty(),
            Some(entries) => {
                let mut weights = BTreeMap::new();
                for (out, _, p) in entries {
                    *weights.entry(out.clone()).or_insert(0.0) += p;
                }
                let total: f64 = weights.values().sum();
                SubDistribution::from_weights(weights).unwrap_or_else(|_| {
                    SubDistribution::empty()
                })
            }
        }
    }

    /// Get the next-state distribution for a given input (marginalizing over outputs).
    pub fn next_state_distribution(&self, input: &Word) -> SubDistribution<X> {
        match self.transitions.get(input) {
            None => SubDistribution::empty(),
            Some(entries) => {
                let mut weights = BTreeMap::new();
                for (_, state, p) in entries {
                    *weights.entry(state.clone()).or_insert(0.0) += p;
                }
                SubDistribution::from_weights(weights).unwrap_or_else(|_| {
                    SubDistribution::empty()
                })
            }
        }
    }

    /// Get the joint distribution over (output, next_state) for a given input.
    pub fn joint_distribution(&self, input: &Word) -> SubDistribution<(OutputSymbol, X)>
    where
        OutputSymbol: Ord,
    {
        match self.transitions.get(input) {
            None => SubDistribution::empty(),
            Some(entries) => {
                let mut weights = BTreeMap::new();
                for (out, state, p) in entries {
                    *weights
                        .entry((out.clone(), state.clone()))
                        .or_insert(0.0) += p;
                }
                SubDistribution::from_weights(weights).unwrap_or_else(|_| {
                    SubDistribution::empty()
                })
            }
        }
    }

    /// Apply a state mapping (functorial action).
    pub fn fmap<Y, F>(&self, f: &F) -> SimpleBehavioralValue<Y>
    where
        Y: Clone + Eq + Hash + Ord + fmt::Debug,
        F: Fn(&X) -> Y,
    {
        let mut result = SimpleBehavioralValue::new();
        for (input, entries) in &self.transitions {
            for (out, state, p) in entries {
                result.add_transition(
                    input.clone(),
                    out.clone(),
                    f(state),
                    *p,
                );
            }
        }
        result
    }

    /// Validate that all transitions form valid sub-distributions.
    pub fn validate(&self, tolerance: f64) -> bool {
        for entries in self.transitions.values() {
            let total: f64 = entries.iter().map(|(_, _, p)| p).sum();
            if total > 1.0 + tolerance {
                return false;
            }
            if entries.iter().any(|(_, _, p)| *p < -tolerance) {
                return false;
            }
        }
        true
    }

    /// Total number of transition entries.
    pub fn total_entries(&self) -> usize {
        self.transitions.values().map(|v| v.len()).sum()
    }

    /// Get all output symbols used.
    pub fn output_symbols(&self) -> HashSet<&OutputSymbol> {
        self.transitions
            .values()
            .flat_map(|v| v.iter().map(|(o, _, _)| o))
            .collect()
    }

    /// Get all next states referenced.
    pub fn next_states(&self) -> HashSet<&X> {
        self.transitions
            .values()
            .flat_map(|v| v.iter().map(|(_, s, _)| s))
            .collect()
    }
}

impl<X: Clone + Eq + Hash + Ord + fmt::Debug> Default for SimpleBehavioralValue<X> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BehavioralFunctor
// ---------------------------------------------------------------------------

/// The behavioral functor F_LLM for LLM coalgebras.
///
/// F_LLM(X) = (Σ_≤k × D(X))^{Σ*_≤n}
///
/// Parameters:
/// - k: size of output alphabet (Σ_≤k)
/// - n: maximum input word length
/// - output_alphabet: the concrete output alphabet
/// - input_alphabet: the concrete input alphabet
#[derive(Debug, Clone)]
pub struct BehavioralFunctor {
    pub output_alphabet: Vec<OutputSymbol>,
    pub input_alphabet: Vec<Symbol>,
    pub max_input_length: usize,
    pub max_output_size: usize,
}

impl BehavioralFunctor {
    pub fn new(
        input_alphabet: Vec<Symbol>,
        output_alphabet: Vec<OutputSymbol>,
        max_input_length: usize,
    ) -> Self {
        let max_output_size = output_alphabet.len();
        Self {
            output_alphabet,
            input_alphabet,
            max_input_length,
            max_output_size,
        }
    }

    /// Get the input words (Σ*_≤n).
    pub fn input_words(&self) -> Vec<Word> {
        Word::enumerate_up_to(&self.input_alphabet, self.max_input_length)
    }

    /// Number of input words.
    pub fn num_input_words(&self) -> usize {
        let a = self.input_alphabet.len();
        let mut total = 0;
        let mut power = 1;
        for _ in 0..=self.max_input_length {
            total += power;
            power *= a;
        }
        total
    }

    /// Apply the functor to a morphism f: X → Y.
    pub fn fmap_behavioral<X, Y, F>(
        value: &SimpleBehavioralValue<X>,
        f: &F,
    ) -> SimpleBehavioralValue<Y>
    where
        X: Clone + Eq + Hash + Ord + fmt::Debug,
        Y: Clone + Eq + Hash + Ord + fmt::Debug,
        F: Fn(&X) -> Y,
    {
        value.fmap(f)
    }

    /// Construct an empty behavioral value.
    pub fn empty_value<X: Clone + Eq + Hash + Ord + fmt::Debug>(&self) -> SimpleBehavioralValue<X> {
        SimpleBehavioralValue::new()
    }

    /// Construct a deterministic behavioral value: for each input, a single
    /// (output, next_state) pair with probability 1.
    pub fn deterministic_value<X: Clone + Eq + Hash + Ord + fmt::Debug>(
        &self,
        mapping: BTreeMap<Word, (OutputSymbol, X)>,
    ) -> SimpleBehavioralValue<X> {
        let mut value = SimpleBehavioralValue::new();
        for (input, (output, state)) in mapping {
            value.add_transition(input, output, state, 1.0);
        }
        value
    }

    /// Compute the behavioral distance between two F_LLM values.
    /// Uses a weighted combination of output distances and continuation distances.
    pub fn behavioral_distance<X: Clone + Eq + Hash + Ord + fmt::Debug>(
        &self,
        v1: &SimpleBehavioralValue<X>,
        v2: &SimpleBehavioralValue<X>,
        state_distance: &dyn Fn(&X, &X) -> f64,
    ) -> f64 {
        let words = self.input_words();
        if words.is_empty() {
            return 0.0;
        }

        let mut max_distance = 0.0f64;

        for word in &words {
            let d1 = v1.joint_distribution(word);
            let d2 = v2.joint_distribution(word);

            // Compute Kantorovich-like distance with a compound ground metric
            let dist = d1.wasserstein_with_metric(&d2, |pair1, pair2| {
                let output_dist = if pair1.0 == pair2.0 { 0.0 } else { 1.0 };
                let state_dist = state_distance(&pair1.1, &pair2.1);
                output_dist + state_dist
            });

            max_distance = max_distance.max(dist);
        }

        max_distance
    }

    /// Compute the behavioral distance using only the output distributions
    /// (ignoring next-state distributions). This is a simpler, faster metric.
    pub fn output_distance(
        &self,
        v1: &SimpleBehavioralValue<StateId>,
        v2: &SimpleBehavioralValue<StateId>,
    ) -> f64 {
        let words = self.input_words();
        if words.is_empty() {
            return 0.0;
        }

        let mut max_distance = 0.0f64;
        for word in &words {
            let d1 = v1.output_distribution(word);
            let d2 = v2.output_distribution(word);
            let tv = d1.total_variation(&d2);
            max_distance = max_distance.max(tv);
        }
        max_distance
    }
}

// ---------------------------------------------------------------------------
// Predicate liftings
// ---------------------------------------------------------------------------

/// A predicate lifting λ for a functor F transforms predicates on X
/// into predicates on F(X).
#[derive(Debug, Clone)]
pub enum PredicateLifting {
    /// Diamond (existential): ◇_a P holds in F(X) if there exists a
    /// transition on input a leading to a state satisfying P.
    Diamond { input: Word },

    /// Box (universal): □_a P holds in F(X) if all transitions on input a
    /// lead to states satisfying P.
    Box { input: Word },

    /// Probabilistic threshold: P^{≥p}_a holds if the probability of
    /// transitioning to a state satisfying P on input a is ≥ p.
    ProbabilisticThreshold { input: Word, threshold: f64 },

    /// Output match: holds if the output on input a matches the given symbol.
    OutputMatch { input: Word, output: OutputSymbol },

    /// Conjunction of liftings.
    Conjunction(Vec<PredicateLifting>),

    /// Disjunction of liftings.
    Disjunction(Vec<PredicateLifting>),

    /// Negation.
    Negation(Box<PredicateLifting>),
}

impl PredicateLifting {
    /// Evaluate the predicate lifting on a behavioral value.
    pub fn evaluate(
        &self,
        value: &SimpleBehavioralValue<StateId>,
        state_predicate: &dyn Fn(&StateId) -> bool,
    ) -> bool {
        match self {
            PredicateLifting::Diamond { input } => {
                if let Some(entries) = value.get_transitions(input) {
                    entries.iter().any(|(_, state, p)| *p > 0.0 && state_predicate(state))
                } else {
                    false
                }
            }
            PredicateLifting::Box { input } => {
                if let Some(entries) = value.get_transitions(input) {
                    entries.iter().all(|(_, state, p)| *p <= 0.0 || state_predicate(state))
                } else {
                    true // vacuously true
                }
            }
            PredicateLifting::ProbabilisticThreshold { input, threshold } => {
                if let Some(entries) = value.get_transitions(input) {
                    let prob: f64 = entries
                        .iter()
                        .filter(|(_, state, _)| state_predicate(state))
                        .map(|(_, _, p)| p)
                        .sum();
                    prob >= *threshold
                } else {
                    *threshold <= 0.0
                }
            }
            PredicateLifting::OutputMatch { input, output } => {
                if let Some(entries) = value.get_transitions(input) {
                    entries.iter().any(|(o, _, p)| *p > 0.0 && o == output)
                } else {
                    false
                }
            }
            PredicateLifting::Conjunction(liftings) => {
                liftings.iter().all(|l| l.evaluate(value, state_predicate))
            }
            PredicateLifting::Disjunction(liftings) => {
                liftings.iter().any(|l| l.evaluate(value, state_predicate))
            }
            PredicateLifting::Negation(inner) => {
                !inner.evaluate(value, state_predicate)
            }
        }
    }

    /// Compute the quantitative version (probability) of the predicate.
    pub fn quantitative_evaluation(
        &self,
        value: &SimpleBehavioralValue<StateId>,
        state_measure: &dyn Fn(&StateId) -> f64,
    ) -> f64 {
        match self {
            PredicateLifting::Diamond { input } => {
                if let Some(entries) = value.get_transitions(input) {
                    entries
                        .iter()
                        .map(|(_, state, p)| p * state_measure(state))
                        .fold(0.0f64, f64::max)
                } else {
                    0.0
                }
            }
            PredicateLifting::Box { input } => {
                if let Some(entries) = value.get_transitions(input) {
                    entries
                        .iter()
                        .map(|(_, state, p)| {
                            if *p > 0.0 {
                                state_measure(state)
                            } else {
                                1.0
                            }
                        })
                        .fold(1.0f64, f64::min)
                } else {
                    1.0
                }
            }
            PredicateLifting::ProbabilisticThreshold { input, .. } => {
                if let Some(entries) = value.get_transitions(input) {
                    entries
                        .iter()
                        .map(|(_, state, p)| p * state_measure(state))
                        .sum()
                } else {
                    0.0
                }
            }
            PredicateLifting::OutputMatch { input, output } => {
                if let Some(entries) = value.get_transitions(input) {
                    entries
                        .iter()
                        .filter(|(o, _, _)| o == output)
                        .map(|(_, _, p)| p)
                        .sum()
                } else {
                    0.0
                }
            }
            PredicateLifting::Conjunction(liftings) => {
                liftings
                    .iter()
                    .map(|l| l.quantitative_evaluation(value, state_measure))
                    .fold(1.0f64, f64::min)
            }
            PredicateLifting::Disjunction(liftings) => {
                liftings
                    .iter()
                    .map(|l| l.quantitative_evaluation(value, state_measure))
                    .fold(0.0f64, f64::max)
            }
            PredicateLifting::Negation(inner) => {
                1.0 - inner.quantitative_evaluation(value, state_measure)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FunctorComposition
// ---------------------------------------------------------------------------

/// Composition of two functors: (G ∘ F)(X) = G(F(X)).
#[derive(Debug, Clone)]
pub struct ComposedFunctor<F1, F2> {
    pub inner: F1,
    pub outer: F2,
}

/// Product functor: (F × G)(X) = F(X) × G(X).
#[derive(Debug, Clone)]
pub struct ProductFunctor<F1, F2> {
    pub first: F1,
    pub second: F2,
}

/// Coproduct functor: (F + G)(X) = F(X) + G(X).
#[derive(Debug, Clone)]
pub struct CoproductFunctor<F1, F2> {
    pub left: F1,
    pub right: F2,
}

/// Constant functor: K_A(X) = A for all X.
#[derive(Debug, Clone)]
pub struct ConstantFunctor<A: Clone + fmt::Debug> {
    pub value: A,
}

/// Identity functor: Id(X) = X.
#[derive(Debug, Clone)]
pub struct IdentityFunctor;

impl Functor for IdentityFunctor {
    type Applied<X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static> = X;

    fn fmap<X, Y, F>(
        fx: &X,
        f: &F,
    ) -> Y
    where
        X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        Y: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        F: Fn(&X) -> Y,
    {
        f(fx)
    }

    fn name(&self) -> &str {
        "Id"
    }
}

/// Powerset functor: P(X) = 2^X (finite subsets).
#[derive(Debug, Clone)]
pub struct PowersetFunctor;

impl Functor for PowersetFunctor {
    type Applied<X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static> = HashSet<X>;

    fn fmap<X, Y, F>(
        fx: &HashSet<X>,
        f: &F,
    ) -> HashSet<Y>
    where
        X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        Y: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        F: Fn(&X) -> Y,
    {
        fx.iter().map(|x| f(x)).collect()
    }

    fn name(&self) -> &str {
        "P (Powerset)"
    }
}

/// Multiset functor: M(X) = multisets over X.
#[derive(Debug, Clone)]
pub struct MultisetFunctor;

impl Functor for MultisetFunctor {
    type Applied<X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static> = BTreeMap<X, usize>;

    fn fmap<X, Y, F>(
        fx: &BTreeMap<X, usize>,
        f: &F,
    ) -> BTreeMap<Y, usize>
    where
        X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        Y: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        F: Fn(&X) -> Y,
    {
        let mut result = BTreeMap::new();
        for (x, &count) in fx {
            *result.entry(f(x)).or_insert(0) += count;
        }
        result
    }

    fn name(&self) -> &str {
        "M (Multiset)"
    }
}

/// List functor: L(X) = X*.
#[derive(Debug, Clone)]
pub struct ListFunctor;

impl Functor for ListFunctor {
    type Applied<X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static> = Vec<X>;

    fn fmap<X, Y, F>(
        fx: &Vec<X>,
        f: &F,
    ) -> Vec<Y>
    where
        X: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        Y: Clone + Eq + Hash + Ord + Send + Sync + fmt::Debug + 'static,
        F: Fn(&X) -> Y,
    {
        fx.iter().map(|x| f(x)).collect()
    }

    fn name(&self) -> &str {
        "L (List)"
    }
}

// ---------------------------------------------------------------------------
// Natural transformations
// ---------------------------------------------------------------------------

/// A natural transformation η: F ⇒ G is a family of morphisms
/// η_X: F(X) → G(X) that commutes with fmap.
#[derive(Debug, Clone)]
pub struct NaturalTransformation {
    pub name: String,
    pub source_functor: String,
    pub target_functor: String,
}

/// Concrete natural transformation from SubDistribution to Powerset (support).
pub fn support_transformation<X: Clone + Eq + Hash + Ord>(
    dist: &SubDistribution<X>,
) -> HashSet<X> {
    dist.support().into_iter().cloned().collect()
}

/// Natural transformation from SubDistribution to the constant functor [0,1] (total mass).
pub fn mass_transformation<X: Clone + Eq + Hash + Ord>(
    dist: &SubDistribution<X>,
) -> f64 {
    dist.total_mass()
}

/// Natural transformation from List to SubDistribution (empirical distribution).
pub fn empirical_transformation<X: Clone + Eq + Hash + Ord>(
    list: &[X],
) -> SubDistribution<X> {
    super::distribution::empirical(list)
}

/// Natural transformation from SubDistribution to List (sorted by weight).
pub fn sorted_support_transformation<X: Clone + Eq + Hash + Ord + fmt::Display>(
    dist: &SubDistribution<X>,
) -> Vec<(X, f64)> {
    let mut items: Vec<(X, f64)> = dist.iter().map(|(k, v)| (k.clone(), v)).collect();
    items.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    items
}

// ---------------------------------------------------------------------------
// Functor lattice
// ---------------------------------------------------------------------------

/// A lattice of functors ordered by "information content".
/// Used to navigate the abstraction hierarchy.
#[derive(Debug, Clone)]
pub struct FunctorLattice {
    pub functors: Vec<FunctorSpec>,
    pub ordering: Vec<(usize, usize)>, // (lower, upper) pairs
}

/// Specification of a functor in the lattice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctorSpec {
    pub name: String,
    pub output_alphabet_size: usize,
    pub max_input_length: usize,
    pub description: String,
}

impl FunctorLattice {
    pub fn new() -> Self {
        Self {
            functors: Vec::new(),
            ordering: Vec::new(),
        }
    }

    pub fn add_functor(&mut self, spec: FunctorSpec) -> usize {
        let idx = self.functors.len();
        self.functors.push(spec);
        idx
    }

    pub fn add_ordering(&mut self, lower: usize, upper: usize) {
        self.ordering.push((lower, upper));
    }

    /// Check if functor at index `lower` is below `upper` in the lattice.
    pub fn is_below(&self, lower: usize, upper: usize) -> bool {
        if lower == upper {
            return true;
        }
        // BFS through ordering
        let mut visited = HashSet::new();
        let mut queue = vec![lower];
        while let Some(current) = queue.pop() {
            if current == upper {
                return true;
            }
            if !visited.insert(current) {
                continue;
            }
            for &(lo, hi) in &self.ordering {
                if lo == current && !visited.contains(&hi) {
                    queue.push(hi);
                }
            }
        }
        false
    }

    /// Find the meet (greatest lower bound) of two functors.
    pub fn meet(&self, a: usize, b: usize) -> Option<usize> {
        let a_below: HashSet<usize> = (0..self.functors.len())
            .filter(|&i| self.is_below(i, a))
            .collect();
        let b_below: HashSet<usize> = (0..self.functors.len())
            .filter(|&i| self.is_below(i, b))
            .collect();
        let common: HashSet<usize> = a_below.intersection(&b_below).copied().collect();
        // Find maximum in common
        common
            .iter()
            .copied()
            .filter(|&c| !common.iter().any(|&d| d != c && self.is_below(c, d)))
            .next()
    }

    /// Find the join (least upper bound) of two functors.
    pub fn join(&self, a: usize, b: usize) -> Option<usize> {
        let a_above: HashSet<usize> = (0..self.functors.len())
            .filter(|&i| self.is_below(a, i))
            .collect();
        let b_above: HashSet<usize> = (0..self.functors.len())
            .filter(|&i| self.is_below(b, i))
            .collect();
        let common: HashSet<usize> = a_above.intersection(&b_above).copied().collect();
        // Find minimum in common
        common
            .iter()
            .copied()
            .filter(|&c| !common.iter().any(|&d| d != c && self.is_below(d, c)))
            .next()
    }

    /// Get all functors reachable from a given starting point (going up).
    pub fn upper_set(&self, start: usize) -> HashSet<usize> {
        (0..self.functors.len())
            .filter(|&i| self.is_below(start, i))
            .collect()
    }

    /// Get all functors below a given point.
    pub fn lower_set(&self, start: usize) -> HashSet<usize> {
        (0..self.functors.len())
            .filter(|&i| self.is_below(i, start))
            .collect()
    }

    /// Build the standard CABER functor lattice with progressively finer functors.
    pub fn standard_lattice(
        alphabet_sizes: &[usize],
        input_lengths: &[usize],
    ) -> Self {
        let mut lattice = Self::new();
        let mut indices = Vec::new();

        for &k in alphabet_sizes {
            for &n in input_lengths {
                let idx = lattice.add_functor(FunctorSpec {
                    name: format!("F_({},{})", k, n),
                    output_alphabet_size: k,
                    max_input_length: n,
                    description: format!(
                        "Behavioral functor with output alphabet size {} and max input length {}",
                        k, n
                    ),
                });
                indices.push((k, n, idx));
            }
        }

        // Add ordering: (k1, n1) ≤ (k2, n2) iff k1 ≤ k2 and n1 ≤ n2
        for i in 0..indices.len() {
            for j in 0..indices.len() {
                if i != j {
                    let (k1, n1, idx1) = indices[i];
                    let (k2, n2, idx2) = indices[j];
                    if k1 <= k2 && n1 <= n2 {
                        lattice.add_ordering(idx1, idx2);
                    }
                }
            }
        }

        lattice
    }
}

impl Default for FunctorLattice {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pushforward maps
// ---------------------------------------------------------------------------

/// Pushforward of a function through the behavioral functor.
/// Given f: X → Y, compute F_LLM(f): F_LLM(X) → F_LLM(Y).
pub fn pushforward_behavioral<X, Y, F>(
    value: &SimpleBehavioralValue<X>,
    f: &F,
) -> SimpleBehavioralValue<Y>
where
    X: Clone + Eq + Hash + Ord + fmt::Debug,
    Y: Clone + Eq + Hash + Ord + fmt::Debug,
    F: Fn(&X) -> Y,
{
    value.fmap(f)
}

/// Pullback of a function: given g: Y → X and a predicate on F_LLM(X),
/// produce a predicate on F_LLM(Y).
pub fn pullback_predicate(
    lifting: &PredicateLifting,
    state_map: &dyn Fn(&StateId) -> StateId,
) -> PredicateLifting {
    // The pullback transforms state references in the predicate
    match lifting {
        PredicateLifting::Diamond { input } => PredicateLifting::Diamond {
            input: input.clone(),
        },
        PredicateLifting::Box { input } => PredicateLifting::Box {
            input: input.clone(),
        },
        PredicateLifting::ProbabilisticThreshold { input, threshold } => {
            PredicateLifting::ProbabilisticThreshold {
                input: input.clone(),
                threshold: *threshold,
            }
        }
        PredicateLifting::OutputMatch { input, output } => PredicateLifting::OutputMatch {
            input: input.clone(),
            output: output.clone(),
        },
        PredicateLifting::Conjunction(liftings) => PredicateLifting::Conjunction(
            liftings
                .iter()
                .map(|l| pullback_predicate(l, state_map))
                .collect(),
        ),
        PredicateLifting::Disjunction(liftings) => PredicateLifting::Disjunction(
            liftings
                .iter()
                .map(|l| pullback_predicate(l, state_map))
                .collect(),
        ),
        PredicateLifting::Negation(inner) => {
            PredicateLifting::Negation(Box::new(pullback_predicate(inner, state_map)))
        }
    }
}

/// Compute the pushforward distance: given distances on X, compute
/// the induced distance on F_LLM(X) values.
pub fn pushforward_distance<X: Clone + Eq + Hash + Ord + fmt::Debug>(
    v1: &SimpleBehavioralValue<X>,
    v2: &SimpleBehavioralValue<X>,
    state_distance: &dyn Fn(&X, &X) -> f64,
    input_words: &[Word],
) -> f64 {
    let mut max_dist = 0.0f64;

    for word in input_words {
        let d1 = v1.joint_distribution(word);
        let d2 = v2.joint_distribution(word);

        let dist = d1.wasserstein_with_metric(&d2, |pair1, pair2| {
            let out_dist = if pair1.0 == pair2.0 { 0.0 } else { 1.0 };
            let state_dist = state_distance(&pair1.1, &pair2.1);
            (out_dist + state_dist) / 2.0
        });

        max_dist = max_dist.max(dist);
    }

    max_dist
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- SubDistributionFunctor tests ---

    #[test]
    fn test_subdist_functor_identity() {
        let dist = SubDistribution::uniform(vec![1u32, 2, 3]);
        let result = SubDistributionFunctor::fmap(&dist, &|x: &u32| *x);
        assert!((result.weight(&1) - dist.weight(&1)).abs() < 1e-10);
        assert!((result.weight(&2) - dist.weight(&2)).abs() < 1e-10);
        assert!((result.weight(&3) - dist.weight(&3)).abs() < 1e-10);
    }

    #[test]
    fn test_subdist_functor_composition() {
        let dist = SubDistribution::uniform(vec![1u32, 2, 3, 4]);
        let f = |x: &u32| x + 10;
        let g = |x: &u32| x * 2;

        // F(g ∘ f) should equal F(g) ∘ F(f)
        let composed_direct = SubDistributionFunctor::fmap(&dist, &|x: &u32| g(&f(x)));
        let composed_sequential = {
            let intermediate = SubDistributionFunctor::fmap(&dist, &f);
            SubDistributionFunctor::fmap(&intermediate, &g)
        };

        for k in [22, 24, 26, 28] {
            assert!(
                (composed_direct.weight(&k) - composed_sequential.weight(&k)).abs() < 1e-10,
                "Functor composition law violated at key {}",
                k
            );
        }
    }

    #[test]
    fn test_subdist_functor_pushforward() {
        let dist = SubDistribution::uniform(vec![1u32, 2, 3, 4]);
        // Map to parity
        let result = SubDistributionFunctor::fmap(&dist, &|x: &u32| x % 2);
        assert!((result.weight(&0) - 0.5).abs() < 1e-10);
        assert!((result.weight(&1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_subdist_unit() {
        let u = SubDistributionFunctor::unit(42u32);
        assert!((u.weight(&42) - 1.0).abs() < 1e-10);
        assert_eq!(u.support_size(), 1);
    }

    #[test]
    fn test_subdist_join() {
        // Distribution over {point(1) with weight 0.5, point(2) with weight 0.5}
        let inner1 = SubDistribution::point(1u32);
        let inner2 = SubDistribution::point(2u32);

        let mut outer_weights = BTreeMap::new();
        outer_weights.insert(inner1, 0.5);
        outer_weights.insert(inner2, 0.5);
        let outer = SubDistribution::from_weights(outer_weights).unwrap();

        let joined = SubDistributionFunctor::join(&outer);
        assert!((joined.weight(&1) - 0.5).abs() < 1e-10);
        assert!((joined.weight(&2) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_strength() {
        let dy = SubDistribution::uniform(vec![10u32, 20]);
        let result = SubDistributionFunctor::strength(&1u32, &dy);
        assert!((result.weight(&(1, 10)) - 0.5).abs() < 1e-10);
        assert!((result.weight(&(1, 20)) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_double_strength() {
        let dx = SubDistribution::uniform(vec![1u32, 2]);
        let dy = SubDistribution::uniform(vec![10u32, 20]);
        let result = SubDistributionFunctor::double_strength(&dx, &dy);
        assert_eq!(result.support_size(), 4);
        assert!((result.weight(&(1, 10)) - 0.25).abs() < 1e-10);
    }

    // --- BehavioralFunctor tests ---

    #[test]
    fn test_behavioral_value_basic() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["hello"]);
        let output = OutputSymbol::new("world");
        let state = StateId::new("s1");

        bv.add_transition(input.clone(), output.clone(), state.clone(), 0.8);
        bv.add_transition(
            input.clone(),
            OutputSymbol::new("hi"),
            StateId::new("s2"),
            0.2,
        );

        assert_eq!(bv.num_inputs(), 1);
        assert_eq!(bv.total_entries(), 2);
        assert!(bv.validate(1e-6));
    }

    #[test]
    fn test_behavioral_value_output_distribution() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["test"]);

        bv.add_transition(
            input.clone(),
            OutputSymbol::new("a"),
            StateId::new("s1"),
            0.6,
        );
        bv.add_transition(
            input.clone(),
            OutputSymbol::new("b"),
            StateId::new("s2"),
            0.4,
        );

        let out_dist = bv.output_distribution(&input);
        assert!((out_dist.weight(&OutputSymbol::new("a")) - 0.6).abs() < 1e-10);
        assert!((out_dist.weight(&OutputSymbol::new("b")) - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_behavioral_value_next_state_distribution() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["test"]);

        bv.add_transition(input.clone(), OutputSymbol::new("a"), StateId::new("s1"), 0.3);
        bv.add_transition(input.clone(), OutputSymbol::new("b"), StateId::new("s1"), 0.4);
        bv.add_transition(input.clone(), OutputSymbol::new("c"), StateId::new("s2"), 0.3);

        let state_dist = bv.next_state_distribution(&input);
        // s1 gets 0.3 + 0.4 = 0.7
        assert!((state_dist.weight(&StateId::new("s1")) - 0.7).abs() < 1e-10);
        assert!((state_dist.weight(&StateId::new("s2")) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_behavioral_value_fmap() {
        let mut bv = SimpleBehavioralValue::<u32>::new();
        let input = Word::from_str_slice(&["x"]);
        bv.add_transition(input.clone(), OutputSymbol::new("a"), 1, 0.5);
        bv.add_transition(input.clone(), OutputSymbol::new("b"), 2, 0.5);

        let mapped = bv.fmap(&|x: &u32| x * 10);
        let entries = mapped.get_transitions(&input).unwrap();
        assert!(entries.iter().any(|(_, s, _)| *s == 10));
        assert!(entries.iter().any(|(_, s, _)| *s == 20));
    }

    #[test]
    fn test_behavioral_functor_creation() {
        let bf = BehavioralFunctor::new(
            vec![Symbol::new("a"), Symbol::new("b")],
            vec![OutputSymbol::new("x"), OutputSymbol::new("y")],
            2,
        );
        assert_eq!(bf.max_output_size, 2);
        assert_eq!(bf.input_alphabet.len(), 2);
        // Words: ε, a, b, aa, ab, ba, bb = 7
        assert_eq!(bf.num_input_words(), 7);
    }

    #[test]
    fn test_behavioral_functor_output_distance() {
        let bf = BehavioralFunctor::new(
            vec![Symbol::new("a")],
            vec![OutputSymbol::new("x"), OutputSymbol::new("y")],
            1,
        );

        let mut v1 = SimpleBehavioralValue::<StateId>::new();
        let mut v2 = SimpleBehavioralValue::<StateId>::new();

        let input = Word::from_str_slice(&["a"]);
        v1.add_transition(input.clone(), OutputSymbol::new("x"), StateId::new("s0"), 1.0);
        v2.add_transition(input.clone(), OutputSymbol::new("y"), StateId::new("s0"), 1.0);

        let dist = bf.output_distance(&v1, &v2);
        assert!(dist > 0.0);
    }

    // --- Predicate lifting tests ---

    #[test]
    fn test_diamond_lifting() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["go"]);
        bv.add_transition(input.clone(), OutputSymbol::new("ok"), StateId::new("good"), 0.7);
        bv.add_transition(input.clone(), OutputSymbol::new("err"), StateId::new("bad"), 0.3);

        let diamond = PredicateLifting::Diamond { input };
        let is_good = |s: &StateId| s.as_str() == "good";

        assert!(diamond.evaluate(&bv, &is_good));
    }

    #[test]
    fn test_box_lifting() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["go"]);
        bv.add_transition(input.clone(), OutputSymbol::new("ok"), StateId::new("good"), 0.7);
        bv.add_transition(input.clone(), OutputSymbol::new("ok"), StateId::new("also_good"), 0.3);

        let box_lift = PredicateLifting::Box { input };
        let is_good = |s: &StateId| s.as_str().contains("good");

        assert!(box_lift.evaluate(&bv, &is_good));
    }

    #[test]
    fn test_probabilistic_threshold() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["query"]);
        bv.add_transition(input.clone(), OutputSymbol::new("a"), StateId::new("safe"), 0.8);
        bv.add_transition(input.clone(), OutputSymbol::new("b"), StateId::new("unsafe"), 0.2);

        let threshold = PredicateLifting::ProbabilisticThreshold {
            input,
            threshold: 0.7,
        };
        let is_safe = |s: &StateId| s.as_str() == "safe";

        assert!(threshold.evaluate(&bv, &is_safe)); // 0.8 >= 0.7
    }

    #[test]
    fn test_output_match_lifting() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["x"]);
        bv.add_transition(input.clone(), OutputSymbol::new("target"), StateId::new("s0"), 1.0);

        let match_lift = PredicateLifting::OutputMatch {
            input: input.clone(),
            output: OutputSymbol::new("target"),
        };

        assert!(match_lift.evaluate(&bv, &|_| true));

        let no_match = PredicateLifting::OutputMatch {
            input,
            output: OutputSymbol::new("other"),
        };
        assert!(!no_match.evaluate(&bv, &|_| true));
    }

    #[test]
    fn test_conjunction_lifting() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["x"]);
        bv.add_transition(input.clone(), OutputSymbol::new("a"), StateId::new("good"), 1.0);

        let conj = PredicateLifting::Conjunction(vec![
            PredicateLifting::Diamond { input: input.clone() },
            PredicateLifting::OutputMatch {
                input,
                output: OutputSymbol::new("a"),
            },
        ]);

        assert!(conj.evaluate(&bv, &|_| true));
    }

    #[test]
    fn test_quantitative_evaluation() {
        let mut bv = SimpleBehavioralValue::<StateId>::new();
        let input = Word::from_str_slice(&["x"]);
        bv.add_transition(input.clone(), OutputSymbol::new("a"), StateId::new("s1"), 0.6);
        bv.add_transition(input.clone(), OutputSymbol::new("b"), StateId::new("s2"), 0.4);

        let lifting = PredicateLifting::ProbabilisticThreshold {
            input,
            threshold: 0.0,
        };

        let measure = |s: &StateId| if s.as_str() == "s1" { 1.0 } else { 0.0 };
        let result = lifting.quantitative_evaluation(&bv, &measure);
        assert!((result - 0.6).abs() < 1e-10);
    }

    // --- Functor lattice tests ---

    #[test]
    fn test_functor_lattice_ordering() {
        let lattice = FunctorLattice::standard_lattice(&[2, 4], &[1, 2]);
        // F_(2,1) ≤ F_(4,2)
        let f_2_1 = lattice
            .functors
            .iter()
            .position(|f| f.output_alphabet_size == 2 && f.max_input_length == 1)
            .unwrap();
        let f_4_2 = lattice
            .functors
            .iter()
            .position(|f| f.output_alphabet_size == 4 && f.max_input_length == 2)
            .unwrap();
        assert!(lattice.is_below(f_2_1, f_4_2));
        assert!(!lattice.is_below(f_4_2, f_2_1));
    }

    #[test]
    fn test_functor_lattice_meet() {
        let lattice = FunctorLattice::standard_lattice(&[2, 4, 8], &[1, 2, 3]);
        let f_4_2 = lattice
            .functors
            .iter()
            .position(|f| f.output_alphabet_size == 4 && f.max_input_length == 2)
            .unwrap();
        let f_8_1 = lattice
            .functors
            .iter()
            .position(|f| f.output_alphabet_size == 8 && f.max_input_length == 1)
            .unwrap();

        let m = lattice.meet(f_4_2, f_8_1);
        assert!(m.is_some());
        let meet_idx = m.unwrap();
        let meet_spec = &lattice.functors[meet_idx];
        // Meet should have output_size ≤ min(4,8) and input_length ≤ min(2,1)
        assert!(meet_spec.output_alphabet_size <= 4);
        assert!(meet_spec.max_input_length <= 1);
    }

    #[test]
    fn test_functor_lattice_upper_set() {
        let lattice = FunctorLattice::standard_lattice(&[2, 4], &[1, 2]);
        let f_2_1 = lattice
            .functors
            .iter()
            .position(|f| f.output_alphabet_size == 2 && f.max_input_length == 1)
            .unwrap();
        let upper = lattice.upper_set(f_2_1);
        assert!(upper.contains(&f_2_1)); // reflexive
        assert!(upper.len() >= 1);
    }

    // --- Other functor tests ---

    #[test]
    fn test_identity_functor() {
        let x = 42u32;
        let result = IdentityFunctor::fmap(&x, &|v: &u32| v + 1);
        assert_eq!(result, 43);
    }

    #[test]
    fn test_powerset_functor_identity() {
        let s: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
        let result = PowersetFunctor::fmap(&s, &|x: &u32| *x);
        assert_eq!(result, s);
    }

    #[test]
    fn test_powerset_functor_map() {
        let s: HashSet<u32> = vec![1, 2, 3].into_iter().collect();
        let result = PowersetFunctor::fmap(&s, &|x: &u32| x * 2);
        let expected: HashSet<u32> = vec![2, 4, 6].into_iter().collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_multiset_functor() {
        let mut m = BTreeMap::new();
        m.insert(1u32, 2usize);
        m.insert(2, 3);
        m.insert(3, 1);

        let result = MultisetFunctor::fmap(&m, &|x: &u32| x % 2);
        // 1->1 (count 2), 2->0 (count 3), 3->1 (count 1)
        // So 0: 3, 1: 3
        assert_eq!(*result.get(&0).unwrap_or(&0), 3);
        assert_eq!(*result.get(&1).unwrap_or(&0), 3);
    }

    #[test]
    fn test_list_functor() {
        let l = vec![1u32, 2, 3, 4];
        let result = ListFunctor::fmap(&l, &|x: &u32| x * x);
        assert_eq!(result, vec![1, 4, 9, 16]);
    }

    // --- Natural transformation tests ---

    #[test]
    fn test_support_transformation() {
        let dist = SubDistribution::uniform(vec![1u32, 2, 3]);
        let supp = support_transformation(&dist);
        assert_eq!(supp.len(), 3);
        assert!(supp.contains(&1));
    }

    #[test]
    fn test_mass_transformation() {
        let dist = SubDistribution::uniform(vec![1u32, 2, 3]);
        assert!((mass_transformation(&dist) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_transformation() {
        let list = vec![1u32, 1, 2, 3];
        let dist = empirical_transformation(&list);
        assert!((dist.weight(&1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sorted_support_transformation() {
        let mut weights = BTreeMap::new();
        weights.insert(1u32, 0.1);
        weights.insert(2, 0.5);
        weights.insert(3, 0.4);
        let dist = SubDistribution::from_weights(weights).unwrap();
        let sorted = sorted_support_transformation(&dist);
        assert_eq!(sorted[0].0, 2); // highest weight first
    }

    #[test]
    fn test_pushforward_behavioral() {
        let mut bv = SimpleBehavioralValue::<u32>::new();
        let input = Word::from_str_slice(&["x"]);
        bv.add_transition(input.clone(), OutputSymbol::new("a"), 1, 0.5);
        bv.add_transition(input.clone(), OutputSymbol::new("b"), 2, 0.5);

        let mapped = pushforward_behavioral(&bv, &|x: &u32| StateId::indexed("s", *x as usize));
        let entries = mapped.get_transitions(&input).unwrap();
        assert_eq!(entries.len(), 2);
    }
}

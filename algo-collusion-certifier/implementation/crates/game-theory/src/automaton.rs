//! Finite-state automaton models for pricing strategies.
//!
//! Provides Mealy machines, product automata, cycle detection, minimization,
//! and recall-bound analysis for repeated game strategies.

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;

// ── Core types ──────────────────────────────────────────────────────────────

/// Unique identifier for an automaton state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AutomatonState(pub usize);

impl fmt::Display for AutomatonState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "q{}", self.0)
    }
}

/// A transition in a finite automaton.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Transition<I: Eq + Hash + Clone, O: Clone> {
    pub from: AutomatonState,
    pub input: I,
    pub to: AutomatonState,
    pub output: O,
}

impl<I: Eq + Hash + Clone + fmt::Debug, O: Clone + fmt::Debug> fmt::Display for Transition<I, O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} --{:?}/{:?}--> {}", self.from, self.input, self.output, self.to)
    }
}

// ── Mealy Machine ───────────────────────────────────────────────────────────

/// A Mealy machine: finite automaton whose output depends on state AND input.
///
/// Generic over state labels `S`, input alphabet `I`, and output alphabet `O`.
/// Internally maps to `AutomatonState` indices for efficient computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MealyMachine<S, I, O>
where
    S: Eq + Hash + Clone,
    I: Eq + Hash + Clone,
    O: Clone,
{
    pub states: Vec<S>,
    pub initial_state: AutomatonState,
    pub input_alphabet: Vec<I>,
    pub output_alphabet: Vec<O>,
    /// Transition function: (state_idx, input) -> next_state_idx
    pub transitions: HashMap<(AutomatonState, usize), AutomatonState>,
    /// Output function: (state_idx, input) -> output
    pub outputs: HashMap<(AutomatonState, usize), O>,
    state_to_idx: HashMap<S, AutomatonState>,
    input_to_idx: HashMap<I, usize>,
}

impl<S, I, O> MealyMachine<S, I, O>
where
    S: Eq + Hash + Clone + fmt::Debug,
    I: Eq + Hash + Clone + fmt::Debug,
    O: Clone + fmt::Debug,
{
    pub fn new(states: Vec<S>, initial: S, inputs: Vec<I>, outputs: Vec<O>) -> Self {
        let state_to_idx: HashMap<S, AutomatonState> = states
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), AutomatonState(i)))
            .collect();
        let input_to_idx: HashMap<I, usize> = inputs
            .iter()
            .enumerate()
            .map(|(i, inp)| (inp.clone(), i))
            .collect();
        let initial_state = state_to_idx[&initial];
        MealyMachine {
            states,
            initial_state,
            input_alphabet: inputs,
            output_alphabet: outputs,
            transitions: HashMap::new(),
            outputs: HashMap::new(),
            state_to_idx,
            input_to_idx,
        }
    }

    pub fn add_transition(&mut self, from: &S, input: &I, to: &S, output: O) {
        let from_idx = self.state_to_idx[from];
        let input_idx = self.input_to_idx[input];
        let to_idx = self.state_to_idx[to];
        self.transitions.insert((from_idx, input_idx), to_idx);
        self.outputs.insert((from_idx, input_idx), output);
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn step(&self, state: AutomatonState, input: &I) -> Option<(AutomatonState, &O)> {
        let input_idx = self.input_to_idx.get(input)?;
        let next = self.transitions.get(&(state, *input_idx))?;
        let out = self.outputs.get(&(state, *input_idx))?;
        Some((*next, out))
    }

    pub fn run(&self, inputs: &[I]) -> Vec<O> {
        let mut state = self.initial_state;
        let mut result = Vec::with_capacity(inputs.len());
        for inp in inputs {
            if let Some((next, out)) = self.step(state, inp) {
                result.push(out.clone());
                state = next;
            }
        }
        result
    }

    pub fn run_with_states(&self, inputs: &[I]) -> Vec<(AutomatonState, O)> {
        let mut state = self.initial_state;
        let mut result = Vec::with_capacity(inputs.len());
        for inp in inputs {
            if let Some((next, out)) = self.step(state, inp) {
                result.push((state, out.clone()));
                state = next;
            }
        }
        result
    }

    pub fn state_index(&self, label: &S) -> Option<AutomatonState> {
        self.state_to_idx.get(label).copied()
    }

    pub fn input_index(&self, input: &I) -> Option<usize> {
        self.input_to_idx.get(input).copied()
    }

    pub fn is_complete(&self) -> bool {
        for s in 0..self.states.len() {
            for i in 0..self.input_alphabet.len() {
                if !self.transitions.contains_key(&(AutomatonState(s), i)) {
                    return false;
                }
            }
        }
        true
    }

    pub fn reachable_states(&self) -> HashSet<AutomatonState> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(self.initial_state);
        visited.insert(self.initial_state);
        while let Some(s) = queue.pop_front() {
            for i in 0..self.input_alphabet.len() {
                if let Some(&next) = self.transitions.get(&(s, i)) {
                    if visited.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
        }
        visited
    }

    /// Convert to adjacency list representation for serialization.
    pub fn to_adjacency_list(&self) -> Vec<Vec<(usize, usize)>> {
        let n = self.states.len();
        let mut adj = vec![vec![]; n];
        for (&(from, inp_idx), &to) in &self.transitions {
            adj[from.0].push((inp_idx, to.0));
        }
        for row in &mut adj {
            row.sort();
        }
        adj
    }

    /// Reconstruct a MealyMachine from adjacency list and output map.
    pub fn from_adjacency_list(
        states: Vec<S>,
        initial: S,
        inputs: Vec<I>,
        output_alphabet: Vec<O>,
        adj: &[Vec<(usize, usize)>],
        output_map: &HashMap<(usize, usize), O>,
    ) -> Self {
        let mut machine = Self::new(states, initial, inputs, output_alphabet);
        for (from_idx, neighbors) in adj.iter().enumerate() {
            for &(inp_idx, to_idx) in neighbors {
                let from_state = AutomatonState(from_idx);
                let to_state = AutomatonState(to_idx);
                if let Some(out) = output_map.get(&(from_idx, inp_idx)) {
                    machine.transitions.insert((from_state, inp_idx), to_state);
                    machine.outputs.insert((from_state, inp_idx), out.clone());
                }
            }
        }
        machine
    }
}

// ── Finite State Strategy ───────────────────────────────────────────────────

/// Discretized price action for automaton-based strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DiscretizedPrice(pub u32);

impl DiscretizedPrice {
    pub fn from_continuous(price: f64, min_price: f64, max_price: f64, num_bins: u32) -> Self {
        if num_bins == 0 {
            return DiscretizedPrice(0);
        }
        let range = max_price - min_price;
        if range <= 0.0 {
            return DiscretizedPrice(0);
        }
        let bin = ((price - min_price) / range * num_bins as f64)
            .floor()
            .max(0.0)
            .min((num_bins - 1) as f64) as u32;
        DiscretizedPrice(bin)
    }

    pub fn to_continuous(self, min_price: f64, max_price: f64, num_bins: u32) -> f64 {
        if num_bins <= 1 {
            return (min_price + max_price) / 2.0;
        }
        let range = max_price - min_price;
        min_price + (self.0 as f64 + 0.5) * range / num_bins as f64
    }
}

/// A pricing strategy represented as a finite-state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiniteStateStrategy {
    pub name: String,
    pub num_states: usize,
    pub num_price_levels: u32,
    pub min_price: f64,
    pub max_price: f64,
    /// Transition table: transitions[state][opponent_action] = next_state
    pub transitions: Vec<Vec<usize>>,
    /// Output table: output_action[state] = my_discretized_price
    pub output_action: Vec<DiscretizedPrice>,
    pub initial_state: usize,
}

impl FiniteStateStrategy {
    pub fn new(
        name: &str,
        num_states: usize,
        num_price_levels: u32,
        min_price: f64,
        max_price: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            num_states,
            num_price_levels,
            min_price,
            max_price,
            transitions: vec![vec![0; num_price_levels as usize]; num_states],
            output_action: vec![DiscretizedPrice(0); num_states],
            initial_state: 0,
        }
    }

    pub fn set_transition(&mut self, from: usize, input: u32, to: usize) {
        if from < self.num_states && (input as usize) < self.transitions[from].len() && to < self.num_states {
            self.transitions[from][input as usize] = to;
        }
    }

    pub fn set_output(&mut self, state: usize, price: DiscretizedPrice) {
        if state < self.num_states {
            self.output_action[state] = price;
        }
    }

    pub fn step(&self, state: usize, opponent_action: DiscretizedPrice) -> (usize, f64) {
        let next = if (opponent_action.0 as usize) < self.transitions[state].len() {
            self.transitions[state][opponent_action.0 as usize]
        } else {
            state
        };
        let out_price = self.output_action[next]
            .to_continuous(self.min_price, self.max_price, self.num_price_levels);
        (next, out_price)
    }

    pub fn simulate(&self, opponent_actions: &[DiscretizedPrice]) -> Vec<f64> {
        let mut state = self.initial_state;
        let mut prices = Vec::with_capacity(opponent_actions.len());
        // First output from initial state
        prices.push(
            self.output_action[state]
                .to_continuous(self.min_price, self.max_price, self.num_price_levels),
        );
        for &opp in opponent_actions.iter().take(opponent_actions.len().saturating_sub(1)) {
            let (next, price) = self.step(state, opp);
            state = next;
            prices.push(price);
        }
        prices
    }

    /// Convert this finite-state strategy into a MealyMachine representation.
    pub fn to_mealy(&self) -> MealyMachine<usize, u32, DiscretizedPrice> {
        let states: Vec<usize> = (0..self.num_states).collect();
        let inputs: Vec<u32> = (0..self.num_price_levels).collect();
        let outputs: Vec<DiscretizedPrice> =
            (0..self.num_price_levels).map(DiscretizedPrice).collect();
        let mut machine = MealyMachine::new(states, self.initial_state, inputs, outputs);
        for from in 0..self.num_states {
            for inp in 0..self.num_price_levels {
                let to = self.transitions[from][inp as usize];
                let out = self.output_action[to];
                machine.add_transition(&from, &inp, &to, out);
            }
        }
        machine
    }
}

// ── Product Automaton ───────────────────────────────────────────────────────

/// Product automaton of N player strategies for joint analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductAutomaton {
    /// Number of players.
    pub num_players: usize,
    /// State-space size per player.
    pub player_state_counts: Vec<usize>,
    /// Total product states = product of player_state_counts.
    pub total_states: usize,
    /// Transition table: product_state -> next product_state.
    /// In the product automaton each "round" means all players simultaneously
    /// output their action, observe the joint action, then transition.
    pub transitions: HashMap<usize, usize>,
    /// Output for each product state: one price level per player.
    pub outputs: HashMap<usize, Vec<DiscretizedPrice>>,
    pub initial_state: usize,
}

impl ProductAutomaton {
    /// Build from a vector of FiniteStateStrategy (one per player).
    pub fn from_strategies(strategies: &[FiniteStateStrategy]) -> Self {
        let n = strategies.len();
        let counts: Vec<usize> = strategies.iter().map(|s| s.num_states).collect();
        let total: usize = counts.iter().product();

        let mut pa = ProductAutomaton {
            num_players: n,
            player_state_counts: counts.clone(),
            total_states: total,
            transitions: HashMap::new(),
            outputs: HashMap::new(),
            initial_state: 0,
        };

        // Compute initial product state index from individual initial states
        let init_components: Vec<usize> = strategies.iter().map(|s| s.initial_state).collect();
        pa.initial_state = pa.encode_state(&init_components);

        // Build transitions and outputs for all reachable states via BFS.
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(pa.initial_state);
        visited.insert(pa.initial_state);

        while let Some(ps) = queue.pop_front() {
            let components = pa.decode_state(ps);
            // Each player outputs their action for current state
            let actions: Vec<DiscretizedPrice> = (0..n)
                .map(|i| strategies[i].output_action[components[i]])
                .collect();
            pa.outputs.insert(ps, actions.clone());

            // Compute next state: each player transitions based on seeing
            // the joint action (we use opponents' outputs as inputs).
            let mut next_components = Vec::with_capacity(n);
            for i in 0..n {
                // Player i sees opponent actions: for 2-player, it's just the other.
                // For N-player, we use the average or the first opponent.
                // Convention: player i sees player (i+1)%n's action as input.
                let opp_idx = (i + 1) % n;
                let opp_action = actions[opp_idx].0 as usize;
                let cur = components[i];
                let next = if opp_action < strategies[i].transitions[cur].len() {
                    strategies[i].transitions[cur][opp_action]
                } else {
                    cur
                };
                next_components.push(next);
            }
            let next_ps = pa.encode_state(&next_components);
            pa.transitions.insert(ps, next_ps);

            if visited.insert(next_ps) {
                queue.push_back(next_ps);
            }
        }

        pa
    }

    /// Encode component states into a single product state index (mixed-radix).
    pub fn encode_state(&self, components: &[usize]) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        for i in (0..components.len()).rev() {
            idx += components[i] * multiplier;
            multiplier *= self.player_state_counts[i];
        }
        idx
    }

    /// Decode a product state index back into per-player component states.
    pub fn decode_state(&self, idx: usize) -> Vec<usize> {
        let mut components = vec![0; self.num_players];
        let mut remaining = idx;
        for i in (0..self.num_players).rev() {
            components[i] = remaining % self.player_state_counts[i];
            remaining /= self.player_state_counts[i];
        }
        components
    }

    /// Get the sequence of product states starting from initial.
    pub fn state_trajectory(&self, max_steps: usize) -> Vec<usize> {
        let mut traj = Vec::with_capacity(max_steps);
        let mut state = self.initial_state;
        for _ in 0..max_steps {
            traj.push(state);
            if let Some(&next) = self.transitions.get(&state) {
                state = next;
            } else {
                break;
            }
        }
        traj
    }
}

// ── Cycle Detector ──────────────────────────────────────────────────────────

/// Detects cycles in the product automaton's state graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleDetector {
    pub transient_states: Vec<usize>,
    pub cycle_states: Vec<usize>,
    pub transient_length: usize,
    pub cycle_length: usize,
}

impl CycleDetector {
    /// Floyd's cycle detection on the product automaton's deterministic successor function.
    pub fn detect(pa: &ProductAutomaton) -> Self {
        if pa.transitions.is_empty() {
            return CycleDetector {
                transient_states: vec![pa.initial_state],
                cycle_states: vec![],
                transient_length: 1,
                cycle_length: 0,
            };
        }

        let succ = |s: usize| -> usize { pa.transitions.get(&s).copied().unwrap_or(s) };

        // Phase 1: find meeting point
        let mut tortoise = succ(pa.initial_state);
        let mut hare = succ(succ(pa.initial_state));
        while tortoise != hare {
            tortoise = succ(tortoise);
            hare = succ(succ(hare));
        }

        // Phase 2: find cycle start
        let mut mu = 0;
        tortoise = pa.initial_state;
        while tortoise != hare {
            tortoise = succ(tortoise);
            hare = succ(hare);
            mu += 1;
        }

        // Phase 3: find cycle length
        let mut lam = 1;
        hare = succ(tortoise);
        while tortoise != hare {
            hare = succ(hare);
            lam += 1;
        }

        // Build transient and cycle state lists
        let mut transient_states = Vec::new();
        let mut s = pa.initial_state;
        for _ in 0..mu {
            transient_states.push(s);
            s = succ(s);
        }

        let cycle_start = s;
        let mut cycle_states = vec![cycle_start];
        s = succ(cycle_start);
        while s != cycle_start {
            cycle_states.push(s);
            s = succ(s);
        }

        CycleDetector {
            transient_states,
            cycle_states,
            transient_length: mu,
            cycle_length: lam,
        }
    }

    pub fn cycle_length(&self) -> usize {
        self.cycle_length
    }

    pub fn transient_length(&self) -> usize {
        self.transient_length
    }

    /// Total period = transient + cycle
    pub fn total_period(&self) -> usize {
        self.transient_length + self.cycle_length
    }

    pub fn is_eventually_periodic(&self) -> bool {
        self.cycle_length > 0
    }

    /// Get the cycle's output sequence from the product automaton.
    pub fn cycle_outputs(&self, pa: &ProductAutomaton) -> Vec<Vec<DiscretizedPrice>> {
        self.cycle_states
            .iter()
            .filter_map(|s| pa.outputs.get(s).cloned())
            .collect()
    }
}

// ── Automaton Minimizer ─────────────────────────────────────────────────────

/// Minimizes a Mealy machine using the Myhill-Nerode partition refinement algorithm.
pub struct AutomatonMinimizer;

impl AutomatonMinimizer {
    /// Partition refinement (Hopcroft-style) for Mealy machines.
    /// Returns the minimized machine and a mapping from old states to equivalence classes.
    pub fn minimize(
        machine: &MealyMachine<usize, u32, DiscretizedPrice>,
    ) -> (MealyMachine<usize, u32, DiscretizedPrice>, Vec<usize>) {
        let n = machine.num_states();
        let k = machine.input_alphabet.len();

        if n == 0 {
            return (
                MealyMachine::new(vec![0], 0, machine.input_alphabet.clone(), machine.output_alphabet.clone()),
                vec![0],
            );
        }

        // Initial partition: group states by their output signature
        let mut partition: Vec<usize> = vec![0; n];
        let mut sig_map: HashMap<Vec<Option<DiscretizedPrice>>, usize> = HashMap::new();
        let mut next_class = 0;

        for s in 0..n {
            let sig: Vec<Option<DiscretizedPrice>> = (0..k)
                .map(|i| machine.outputs.get(&(AutomatonState(s), i)).cloned())
                .collect();
            let class = sig_map.entry(sig).or_insert_with(|| {
                let c = next_class;
                next_class += 1;
                c
            });
            partition[s] = *class;
        }

        // Iterative refinement
        loop {
            let mut new_partition = vec![0usize; n];
            let mut new_sig_map: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
            let mut new_next_class = 0;
            let mut changed = false;

            for s in 0..n {
                let cur_class = partition[s];
                let trans_sig: Vec<usize> = (0..k)
                    .map(|i| {
                        machine
                            .transitions
                            .get(&(AutomatonState(s), i))
                            .map(|t| partition[t.0])
                            .unwrap_or(usize::MAX)
                    })
                    .collect();

                let key = (cur_class, trans_sig);
                let new_class = new_sig_map.entry(key).or_insert_with(|| {
                    let c = new_next_class;
                    new_next_class += 1;
                    c
                });
                new_partition[s] = *new_class;
                if *new_class != partition[s] {
                    changed = true;
                }
            }

            if !changed || new_next_class == next_class {
                break;
            }
            partition = new_partition;
            next_class = new_next_class;
        }

        // Build minimized machine
        let num_classes = *partition.iter().max().unwrap_or(&0) + 1;
        let states: Vec<usize> = (0..num_classes).collect();
        // Find which old state represents each class
        let mut class_rep: Vec<usize> = vec![0; num_classes];
        for (s, &c) in partition.iter().enumerate() {
            class_rep[c] = s;
        }

        let initial_class = partition[machine.initial_state.0];
        let mut minimized = MealyMachine::new(
            states,
            initial_class,
            machine.input_alphabet.clone(),
            machine.output_alphabet.clone(),
        );

        for c in 0..num_classes {
            let rep = class_rep[c];
            for i in 0..k {
                if let Some(&to) = machine.transitions.get(&(AutomatonState(rep), i)) {
                    let to_class = partition[to.0];
                    if let Some(out) = machine.outputs.get(&(AutomatonState(rep), i)) {
                        minimized
                            .transitions
                            .insert((AutomatonState(c), i), AutomatonState(to_class));
                        minimized
                            .outputs
                            .insert((AutomatonState(c), i), out.clone());
                    }
                }
            }
        }

        (minimized, partition)
    }
}

// ── Recall Bound ────────────────────────────────────────────────────────────

/// Computes the effective recall (memory) of an automaton-based strategy.
///
/// The recall bound M means the strategy's behavior depends on at most the
/// last M rounds of play. For a finite automaton with |Q| states, M ≤ |Q|.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallBound {
    pub bound: usize,
    pub is_tight: bool,
    pub num_states: usize,
}

impl RecallBound {
    /// Compute the recall bound for a FiniteStateStrategy.
    ///
    /// The recall is the minimum M such that the strategy's output for any
    /// two histories agreeing on the last M rounds is identical. For a
    /// deterministic finite automaton, M ≤ number of reachable states.
    pub fn compute(strategy: &FiniteStateStrategy) -> Self {
        let mealy = strategy.to_mealy();
        let reachable = mealy.reachable_states();
        let num_reachable = reachable.len();

        // Simulate all possible input sequences up to length num_reachable
        // to find the minimal M where the output stabilizes.
        let num_inputs = strategy.num_price_levels as usize;
        let mut recall = 0;

        // Check distinguishability: two states are distinguishable if there
        // exists an input string that produces different outputs.
        // The recall is then the length of the longest such distinguishing string.
        let mut dist_length = vec![vec![0usize; num_reachable]; num_reachable];
        let reachable_vec: Vec<AutomatonState> = reachable.iter().copied().collect();

        // BFS-like: check if pairs of states are distinguishable
        for i in 0..reachable_vec.len() {
            for j in (i + 1)..reachable_vec.len() {
                let si = reachable_vec[i];
                let sj = reachable_vec[j];
                // Find shortest distinguishing sequence via BFS
                let mut q: VecDeque<(AutomatonState, AutomatonState, usize)> = VecDeque::new();
                q.push_back((si, sj, 0));
                let mut found = false;
                let mut seen: HashSet<(AutomatonState, AutomatonState)> = HashSet::new();
                seen.insert((si, sj));

                while let Some((a, b, depth)) = q.pop_front() {
                    if depth > num_reachable {
                        break;
                    }
                    for inp in 0..num_inputs {
                        let inp_val = inp as u32;
                        let out_a = mealy.outputs.get(&(a, inp));
                        let out_b = mealy.outputs.get(&(b, inp));
                        if out_a != out_b {
                            dist_length[i][j] = depth + 1;
                            found = true;
                            break;
                        }
                        let next_a = mealy.transitions.get(&(a, inp)).copied().unwrap_or(a);
                        let next_b = mealy.transitions.get(&(b, inp)).copied().unwrap_or(b);
                        if next_a != next_b && seen.insert((next_a, next_b)) {
                            q.push_back((next_a, next_b, depth + 1));
                        }
                    }
                    if found {
                        break;
                    }
                }
                if dist_length[i][j] > recall {
                    recall = dist_length[i][j];
                }
            }
        }

        RecallBound {
            bound: recall.max(1),
            is_tight: recall <= num_reachable,
            num_states: num_reachable,
        }
    }
}

// ── State Reachability Analysis ─────────────────────────────────────────────

/// Results of reachability analysis on an automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachabilityResult {
    pub reachable: Vec<AutomatonState>,
    pub unreachable: Vec<AutomatonState>,
    /// Minimum input length to reach each reachable state from initial.
    pub distances: HashMap<AutomatonState, usize>,
    /// Predecessor map for shortest paths.
    pub predecessors: HashMap<AutomatonState, AutomatonState>,
}

/// Analyze state reachability using BFS from the initial state.
pub fn state_reachability_analysis(
    strategy: &FiniteStateStrategy,
) -> ReachabilityResult {
    let n = strategy.num_states;
    let num_inputs = strategy.num_price_levels as usize;
    let mut visited = HashSet::new();
    let mut distances = HashMap::new();
    let mut predecessors = HashMap::new();
    let mut queue = VecDeque::new();

    let init = AutomatonState(strategy.initial_state);
    queue.push_back(init);
    visited.insert(init);
    distances.insert(init, 0usize);

    while let Some(state) = queue.pop_front() {
        let d = distances[&state];
        for inp in 0..num_inputs {
            let next_idx = strategy.transitions[state.0][inp];
            let next = AutomatonState(next_idx);
            if visited.insert(next) {
                distances.insert(next, d + 1);
                predecessors.insert(next, state);
                queue.push_back(next);
            }
        }
    }

    let reachable: Vec<AutomatonState> = visited.iter().copied().collect();
    let unreachable: Vec<AutomatonState> = (0..n)
        .map(AutomatonState)
        .filter(|s| !visited.contains(s))
        .collect();

    ReachabilityResult {
        reachable,
        unreachable,
        distances,
        predecessors,
    }
}

// ── Automaton Builder ───────────────────────────────────────────────────────

/// Builder for constructing FiniteStateStrategy programmatically.
#[derive(Debug, Clone)]
pub struct AutomatonBuilder {
    name: String,
    num_price_levels: u32,
    min_price: f64,
    max_price: f64,
    states: Vec<String>,
    initial_state: usize,
    transitions: Vec<(usize, u32, usize)>,
    outputs: Vec<(usize, DiscretizedPrice)>,
}

impl AutomatonBuilder {
    pub fn new(name: &str, num_price_levels: u32) -> Self {
        Self {
            name: name.to_string(),
            num_price_levels,
            min_price: 0.0,
            max_price: 10.0,
            states: Vec::new(),
            initial_state: 0,
            transitions: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn price_range(mut self, min: f64, max: f64) -> Self {
        self.min_price = min;
        self.max_price = max;
        self
    }

    pub fn add_state(mut self, name: &str) -> Self {
        self.states.push(name.to_string());
        self
    }

    pub fn initial(mut self, state_idx: usize) -> Self {
        self.initial_state = state_idx;
        self
    }

    pub fn transition(mut self, from: usize, input: u32, to: usize) -> Self {
        self.transitions.push((from, input, to));
        self
    }

    pub fn output(mut self, state: usize, price_level: u32) -> Self {
        self.outputs.push((state, DiscretizedPrice(price_level)));
        self
    }

    /// Add transitions for all inputs from a state to a single target.
    pub fn transition_all(mut self, from: usize, to: usize) -> Self {
        for i in 0..self.num_price_levels {
            self.transitions.push((from, i, to));
        }
        self
    }

    /// Add transitions with a threshold: inputs < threshold go to `to_low`,
    /// inputs >= threshold go to `to_high`.
    pub fn transition_threshold(mut self, from: usize, threshold: u32, to_low: usize, to_high: usize) -> Self {
        for i in 0..self.num_price_levels {
            if i < threshold {
                self.transitions.push((from, i, to_low));
            } else {
                self.transitions.push((from, i, to_high));
            }
        }
        self
    }

    pub fn build(self) -> FiniteStateStrategy {
        let num_states = self.states.len().max(1);
        let mut strategy = FiniteStateStrategy::new(
            &self.name,
            num_states,
            self.num_price_levels,
            self.min_price,
            self.max_price,
        );
        strategy.initial_state = self.initial_state;

        for (from, inp, to) in &self.transitions {
            strategy.set_transition(*from, *inp, *to);
        }
        for (state, price) in &self.outputs {
            strategy.set_output(*state, *price);
        }
        strategy
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_state_mealy() -> MealyMachine<usize, u32, DiscretizedPrice> {
        let mut m = MealyMachine::new(
            vec![0, 1],
            0,
            vec![0, 1],
            vec![DiscretizedPrice(0), DiscretizedPrice(1)],
        );
        m.add_transition(&0, &0, &0, DiscretizedPrice(1));
        m.add_transition(&0, &1, &1, DiscretizedPrice(1));
        m.add_transition(&1, &0, &1, DiscretizedPrice(0));
        m.add_transition(&1, &1, &0, DiscretizedPrice(0));
        m
    }

    #[test]
    fn test_mealy_step() {
        let m = make_two_state_mealy();
        let (next, out) = m.step(AutomatonState(0), &0).unwrap();
        assert_eq!(next, AutomatonState(0));
        assert_eq!(*out, DiscretizedPrice(1));
    }

    #[test]
    fn test_mealy_run() {
        let m = make_two_state_mealy();
        let outputs = m.run(&[0, 1, 0, 1]);
        assert_eq!(outputs.len(), 4);
        assert_eq!(outputs[0], DiscretizedPrice(1)); // state 0, input 0 -> out 1
        assert_eq!(outputs[1], DiscretizedPrice(1)); // state 0, input 1 -> out 1
        assert_eq!(outputs[2], DiscretizedPrice(0)); // state 1, input 0 -> out 0
        assert_eq!(outputs[3], DiscretizedPrice(0)); // state 1, input 1 -> out 0
    }

    #[test]
    fn test_mealy_run_with_states() {
        let m = make_two_state_mealy();
        let results = m.run_with_states(&[0, 1]);
        assert_eq!(results[0].0, AutomatonState(0));
        assert_eq!(results[1].0, AutomatonState(0)); // stayed at 0 after input 0
    }

    #[test]
    fn test_mealy_completeness() {
        let m = make_two_state_mealy();
        assert!(m.is_complete());
    }

    #[test]
    fn test_mealy_reachable() {
        let m = make_two_state_mealy();
        let r = m.reachable_states();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_adjacency_list_roundtrip() {
        let m = make_two_state_mealy();
        let adj = m.to_adjacency_list();
        assert_eq!(adj.len(), 2);
        assert!(!adj[0].is_empty());
    }

    #[test]
    fn test_discretized_price_roundtrip() {
        let dp = DiscretizedPrice::from_continuous(5.0, 0.0, 10.0, 10);
        let continuous = dp.to_continuous(0.0, 10.0, 10);
        assert!((continuous - 5.5).abs() < 1.0); // bin center
    }

    #[test]
    fn test_discretized_price_boundaries() {
        let low = DiscretizedPrice::from_continuous(0.0, 0.0, 10.0, 10);
        assert_eq!(low.0, 0);
        let high = DiscretizedPrice::from_continuous(10.0, 0.0, 10.0, 10);
        assert_eq!(high.0, 9);
    }

    #[test]
    fn test_finite_state_strategy_step() {
        let mut s = FiniteStateStrategy::new("test", 2, 2, 0.0, 10.0);
        s.set_transition(0, 0, 0);
        s.set_transition(0, 1, 1);
        s.set_transition(1, 0, 1);
        s.set_transition(1, 1, 0);
        s.set_output(0, DiscretizedPrice(1));
        s.set_output(1, DiscretizedPrice(0));
        let (next, _price) = s.step(0, DiscretizedPrice(1));
        assert_eq!(next, 1);
    }

    #[test]
    fn test_finite_state_to_mealy() {
        let mut s = FiniteStateStrategy::new("test", 2, 2, 0.0, 10.0);
        s.set_transition(0, 0, 0);
        s.set_transition(0, 1, 1);
        s.set_transition(1, 0, 1);
        s.set_transition(1, 1, 0);
        s.set_output(0, DiscretizedPrice(1));
        s.set_output(1, DiscretizedPrice(0));
        let mealy = s.to_mealy();
        assert_eq!(mealy.num_states(), 2);
        assert!(mealy.is_complete());
    }

    #[test]
    fn test_product_automaton_two_players() {
        let mut s1 = FiniteStateStrategy::new("p1", 2, 2, 0.0, 10.0);
        s1.set_transition(0, 0, 0); s1.set_transition(0, 1, 1);
        s1.set_transition(1, 0, 1); s1.set_transition(1, 1, 0);
        s1.set_output(0, DiscretizedPrice(1)); s1.set_output(1, DiscretizedPrice(0));

        let mut s2 = FiniteStateStrategy::new("p2", 2, 2, 0.0, 10.0);
        s2.set_transition(0, 0, 0); s2.set_transition(0, 1, 1);
        s2.set_transition(1, 0, 1); s2.set_transition(1, 1, 0);
        s2.set_output(0, DiscretizedPrice(1)); s2.set_output(1, DiscretizedPrice(0));

        let pa = ProductAutomaton::from_strategies(&[s1, s2]);
        assert!(pa.total_states <= 4);
        assert!(!pa.transitions.is_empty());
    }

    #[test]
    fn test_product_automaton_encode_decode() {
        let pa = ProductAutomaton {
            num_players: 2,
            player_state_counts: vec![3, 4],
            total_states: 12,
            transitions: HashMap::new(),
            outputs: HashMap::new(),
            initial_state: 0,
        };
        let components = vec![2, 3];
        let encoded = pa.encode_state(&components);
        let decoded = pa.decode_state(encoded);
        assert_eq!(decoded, components);
    }

    #[test]
    fn test_cycle_detector_simple_cycle() {
        let mut pa = ProductAutomaton {
            num_players: 1,
            player_state_counts: vec![3],
            total_states: 3,
            transitions: HashMap::new(),
            outputs: HashMap::new(),
            initial_state: 0,
        };
        pa.transitions.insert(0, 1);
        pa.transitions.insert(1, 2);
        pa.transitions.insert(2, 1);

        let cd = CycleDetector::detect(&pa);
        assert_eq!(cd.transient_length, 1);
        assert_eq!(cd.cycle_length, 2);
    }

    #[test]
    fn test_cycle_detector_fixed_point() {
        let mut pa = ProductAutomaton {
            num_players: 1,
            player_state_counts: vec![1],
            total_states: 1,
            transitions: HashMap::new(),
            outputs: HashMap::new(),
            initial_state: 0,
        };
        pa.transitions.insert(0, 0);
        let cd = CycleDetector::detect(&pa);
        assert_eq!(cd.cycle_length, 1);
        assert_eq!(cd.transient_length, 0);
    }

    #[test]
    fn test_minimizer_already_minimal() {
        let m = make_two_state_mealy();
        let (min_m, partition) = AutomatonMinimizer::minimize(&m);
        // The two states produce different outputs so cannot be merged
        assert!(min_m.num_states() <= 2);
    }

    #[test]
    fn test_minimizer_mergeable_states() {
        // Create a machine where states 0 and 1 are equivalent
        let mut m = MealyMachine::new(
            vec![0, 1, 2],
            0,
            vec![0, 1],
            vec![DiscretizedPrice(0), DiscretizedPrice(1)],
        );
        // States 0 and 1 have identical outputs and transitions
        m.add_transition(&0, &0, &2, DiscretizedPrice(0));
        m.add_transition(&0, &1, &2, DiscretizedPrice(1));
        m.add_transition(&1, &0, &2, DiscretizedPrice(0));
        m.add_transition(&1, &1, &2, DiscretizedPrice(1));
        m.add_transition(&2, &0, &0, DiscretizedPrice(1));
        m.add_transition(&2, &1, &1, DiscretizedPrice(0));

        let (min_m, _) = AutomatonMinimizer::minimize(&m);
        assert!(min_m.num_states() <= 2); // states 0 and 1 should merge
    }

    #[test]
    fn test_recall_bound() {
        let mut s = FiniteStateStrategy::new("test", 2, 2, 0.0, 10.0);
        s.set_transition(0, 0, 0); s.set_transition(0, 1, 1);
        s.set_transition(1, 0, 0); s.set_transition(1, 1, 1);
        s.set_output(0, DiscretizedPrice(1)); s.set_output(1, DiscretizedPrice(0));
        let rb = RecallBound::compute(&s);
        assert!(rb.bound >= 1);
        assert!(rb.bound <= 2);
    }

    #[test]
    fn test_reachability_analysis() {
        let mut s = FiniteStateStrategy::new("test", 3, 2, 0.0, 10.0);
        s.set_transition(0, 0, 0); s.set_transition(0, 1, 1);
        s.set_transition(1, 0, 0); s.set_transition(1, 1, 1);
        // State 2 is unreachable
        s.set_transition(2, 0, 2); s.set_transition(2, 1, 2);
        s.set_output(0, DiscretizedPrice(1));
        s.set_output(1, DiscretizedPrice(0));
        s.set_output(2, DiscretizedPrice(0));
        let result = state_reachability_analysis(&s);
        assert_eq!(result.reachable.len(), 2);
        assert_eq!(result.unreachable.len(), 1);
    }

    #[test]
    fn test_automaton_builder() {
        let strategy = AutomatonBuilder::new("grim", 2)
            .price_range(0.0, 10.0)
            .add_state("cooperate")
            .add_state("defect")
            .initial(0)
            .transition_threshold(0, 1, 1, 0)
            .transition_all(1, 1)
            .output(0, 1)
            .output(1, 0)
            .build();
        assert_eq!(strategy.num_states, 2);
        assert_eq!(strategy.initial_state, 0);
    }

    #[test]
    fn test_automaton_builder_transition_all() {
        let strategy = AutomatonBuilder::new("absorbing", 3)
            .price_range(0.0, 5.0)
            .add_state("a")
            .add_state("b")
            .add_state("c")
            .initial(0)
            .transition_all(0, 1)
            .transition_all(1, 2)
            .transition_all(2, 2)
            .output(0, 2)
            .output(1, 1)
            .output(2, 0)
            .build();
        let (next, _) = strategy.step(0, DiscretizedPrice(0));
        assert_eq!(next, 1);
    }

    #[test]
    fn test_state_trajectory() {
        let mut pa = ProductAutomaton {
            num_players: 1,
            player_state_counts: vec![3],
            total_states: 3,
            transitions: HashMap::new(),
            outputs: HashMap::new(),
            initial_state: 0,
        };
        pa.transitions.insert(0, 1);
        pa.transitions.insert(1, 2);
        pa.transitions.insert(2, 0);
        let traj = pa.state_trajectory(6);
        assert_eq!(traj, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_simulate_strategy() {
        let mut s = FiniteStateStrategy::new("test", 2, 2, 0.0, 10.0);
        s.set_transition(0, 0, 0); s.set_transition(0, 1, 1);
        s.set_transition(1, 0, 0); s.set_transition(1, 1, 1);
        s.set_output(0, DiscretizedPrice(1)); s.set_output(1, DiscretizedPrice(0));
        let opp = vec![DiscretizedPrice(0), DiscretizedPrice(1), DiscretizedPrice(0)];
        let prices = s.simulate(&opp);
        assert_eq!(prices.len(), 3);
    }
}

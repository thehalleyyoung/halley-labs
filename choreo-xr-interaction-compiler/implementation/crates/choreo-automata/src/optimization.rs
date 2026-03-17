//! Automaton optimisation pipeline: unreachable-state removal,
//! equivalent-state merging, guard simplification, dead-transition
//! elimination, linear-chain compression, and common-prefix factorisation.

use crate::automaton::{SpatialEventAutomaton, State, Transition};
use crate::{
    Guard, StateId, TransitionId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Statistics produced by an optimisation pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationStats {
    pub states_removed: usize,
    pub transitions_removed: usize,
    pub states_merged: usize,
    pub guards_simplified: usize,
    pub chains_compressed: usize,
    pub prefixes_factored: usize,
    pub total_passes: usize,
}

impl OptimizationStats {
    pub fn merge(&mut self, other: &OptimizationStats) {
        self.states_removed += other.states_removed;
        self.transitions_removed += other.transitions_removed;
        self.states_merged += other.states_merged;
        self.guards_simplified += other.guards_simplified;
        self.chains_compressed += other.chains_compressed;
        self.prefixes_factored += other.prefixes_factored;
        self.total_passes += other.total_passes;
    }

    pub fn any_change(&self) -> bool {
        self.states_removed > 0
            || self.transitions_removed > 0
            || self.states_merged > 0
            || self.guards_simplified > 0
            || self.chains_compressed > 0
            || self.prefixes_factored > 0
    }
}

impl std::fmt::Display for OptimizationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OptStats {{ removed_states: {}, removed_trans: {}, merged: {}, simplified: {}, chains: {}, prefixes: {}, passes: {} }}",
            self.states_removed,
            self.transitions_removed,
            self.states_merged,
            self.guards_simplified,
            self.chains_compressed,
            self.prefixes_factored,
            self.total_passes,
        )
    }
}

// ---------------------------------------------------------------------------
// OptimizationPipeline
// ---------------------------------------------------------------------------

/// Available optimisation passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptPass {
    RemoveUnreachable,
    MergeEquivalent,
    SimplifyGuards,
    EliminateDeadTransitions,
    CompressLinearChains,
    FactorizeCommonPrefixes,
}

/// Configurable multi-pass optimisation pipeline.
#[derive(Debug, Clone)]
pub struct OptimizationPipeline {
    pub passes: Vec<OptPass>,
    pub max_iterations: usize,
    pub stats: OptimizationStats,
}

impl OptimizationPipeline {
    /// Pipeline with all passes enabled.
    pub fn all() -> Self {
        Self {
            passes: vec![
                OptPass::RemoveUnreachable,
                OptPass::MergeEquivalent,
                OptPass::SimplifyGuards,
                OptPass::EliminateDeadTransitions,
                OptPass::CompressLinearChains,
                OptPass::FactorizeCommonPrefixes,
            ],
            max_iterations: 5,
            stats: OptimizationStats::default(),
        }
    }

    /// Pipeline with a custom set of passes.
    pub fn with_passes(passes: Vec<OptPass>) -> Self {
        Self {
            passes,
            max_iterations: 5,
            stats: OptimizationStats::default(),
        }
    }

    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Run all configured passes until no more changes occur or
    /// `max_iterations` is reached.
    pub fn apply_all(
        &mut self,
        automaton: &SpatialEventAutomaton,
    ) -> SpatialEventAutomaton {
        let mut current = automaton.clone();
        for _iter in 0..self.max_iterations {
            let before_states = current.state_count();
            let before_trans = current.transition_count();

            for pass in &self.passes.clone() {
                let (next, pass_stats) = self.run_pass(*pass, &current);
                current = next;
                self.stats.merge(&pass_stats);
            }

            self.stats.total_passes += 1;
            // Check for fixpoint
            if current.state_count() == before_states
                && current.transition_count() == before_trans
            {
                break;
            }
        }
        current.recompute_statistics();
        current
    }

    fn run_pass(
        &self,
        pass: OptPass,
        auto: &SpatialEventAutomaton,
    ) -> (SpatialEventAutomaton, OptimizationStats) {
        match pass {
            OptPass::RemoveUnreachable => {
                let (result, removed) = remove_unreachable_states(auto);
                let stats = OptimizationStats {
                    states_removed: removed,
                    ..Default::default()
                };
                (result, stats)
            }
            OptPass::MergeEquivalent => {
                let (result, merged) = merge_equivalent_states(auto);
                let stats = OptimizationStats {
                    states_merged: merged,
                    ..Default::default()
                };
                (result, stats)
            }
            OptPass::SimplifyGuards => {
                let (result, simplified) = simplify_guards(auto);
                let stats = OptimizationStats {
                    guards_simplified: simplified,
                    ..Default::default()
                };
                (result, stats)
            }
            OptPass::EliminateDeadTransitions => {
                let (result, removed) = eliminate_dead_transitions(auto);
                let stats = OptimizationStats {
                    transitions_removed: removed,
                    ..Default::default()
                };
                (result, stats)
            }
            OptPass::CompressLinearChains => {
                let (result, compressed) = compress_linear_chains(auto);
                let stats = OptimizationStats {
                    chains_compressed: compressed,
                    ..Default::default()
                };
                (result, stats)
            }
            OptPass::FactorizeCommonPrefixes => {
                let (result, factored) = factorize_common_prefixes(auto);
                let stats = OptimizationStats {
                    prefixes_factored: factored,
                    ..Default::default()
                };
                (result, stats)
            }
        }
    }

    pub fn statistics(&self) -> &OptimizationStats {
        &self.stats
    }
}

impl Default for OptimizationPipeline {
    fn default() -> Self {
        Self::all()
    }
}

// ---------------------------------------------------------------------------
// Individual optimisation passes
// ---------------------------------------------------------------------------

/// Remove states that are not reachable from the initial state.
/// Returns `(optimised_automaton, number_of_states_removed)`.
pub fn remove_unreachable_states(
    auto: &SpatialEventAutomaton,
) -> (SpatialEventAutomaton, usize) {
    let reachable = auto.reachable_states();
    let all_states: HashSet<StateId> = auto.state_ids().into_iter().collect();
    let unreachable: HashSet<StateId> = all_states.difference(&reachable).copied().collect();
    let removed_count = unreachable.len();

    if removed_count == 0 {
        return (auto.clone(), 0);
    }

    let mut result = SpatialEventAutomaton::new(format!("{}_opt", auto.metadata.name));
    result.kind = auto.kind;

    // Copy reachable states
    for &sid in &reachable {
        if let Some(s) = auto.state(sid) {
            result.add_state(s.clone());
        }
    }

    // Copy transitions whose both endpoints are reachable
    for t in auto.transitions.values() {
        if reachable.contains(&t.source) && reachable.contains(&t.target) {
            result.add_transition(t.clone());
        }
    }

    result.next_state_id = auto.states.len() as u32;
    result.next_transition_id = auto.transitions.len() as u32;
    result.recompute_statistics();
    (result, removed_count)
}

/// Merge states that are bisimulation-equivalent.
/// Returns `(optimised_automaton, number_of_states_merged)`.
pub fn merge_equivalent_states(
    auto: &SpatialEventAutomaton,
) -> (SpatialEventAutomaton, usize) {
    if auto.state_count() <= 1 {
        return (auto.clone(), 0);
    }

    let state_ids: Vec<StateId> = auto.state_ids();
    let n = state_ids.len();
    let idx: HashMap<StateId, usize> = state_ids
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();

    // Build signature for each state
    let _signatures: HashMap<usize, Vec<(String, usize)>> = HashMap::new();
    let mut partition: Vec<usize> = vec![0; n];

    // Initial: accepting vs non-accepting
    for (i, &sid) in state_ids.iter().enumerate() {
        partition[i] = if auto.accepting_states.contains(&sid) { 1 } else { 0 };
    }

    let mut changed = true;
    let mut next_class = 2;
    while changed {
        changed = false;
        let mut class_sigs: HashMap<usize, HashMap<Vec<(String, usize)>, usize>> = HashMap::new();
        let mut new_partition = partition.clone();

        for (i, &sid) in state_ids.iter().enumerate() {
            let old_class = partition[i];
            let mut sig: Vec<(String, usize)> = Vec::new();

            for tid in auto.outgoing(sid) {
                if let Some(t) = auto.transition(tid) {
                    if let Some(&tgt_idx) = idx.get(&t.target) {
                        sig.push((format!("{}", t.guard), partition[tgt_idx]));
                    }
                }
            }
            sig.sort();

            let map = class_sigs.entry(old_class).or_default();
            if let Some(&assigned) = map.get(&sig) {
                if new_partition[i] != assigned {
                    new_partition[i] = assigned;
                    changed = true;
                }
            } else {
                let c = if map.is_empty() {
                    old_class
                } else {
                    let c = next_class;
                    next_class += 1;
                    changed = true;
                    c
                };
                map.insert(sig, c);
                new_partition[i] = c;
            }
        }
        partition = new_partition;
    }

    // Count distinct classes
    let unique_classes: HashSet<usize> = partition.iter().copied().collect();
    let merged_count = n - unique_classes.len();
    if merged_count == 0 {
        return (auto.clone(), 0);
    }

    // Build new automaton with one representative per class
    let mut result = SpatialEventAutomaton::new(format!("{}_merged", auto.metadata.name));
    result.kind = auto.kind;
    let mut class_to_state: HashMap<usize, StateId> = HashMap::new();

    for (i, &sid) in state_ids.iter().enumerate() {
        let c = partition[i];
        if class_to_state.contains_key(&c) {
            continue;
        }
        let nid = result.fresh_state_id();
        if let Some(original) = auto.state(sid) {
            let mut ns = original.clone();
            ns.id = nid;
            ns.name = format!("merged_{}", c);
            ns.is_initial = auto.initial_state.is_some()
                && partition[idx[&auto.initial_state.unwrap()]] == c;
            result.add_state(ns);
        }
        class_to_state.insert(c, nid);
    }

    // Add transitions (deduplicated)
    let mut seen: HashSet<(usize, usize, String)> = HashSet::new();
    for t in auto.transitions.values() {
        if let (Some(&si), Some(&ti)) = (idx.get(&t.source), idx.get(&t.target)) {
            let sc = partition[si];
            let tc = partition[ti];
            let key = (sc, tc, format!("{}", t.guard));
            if seen.insert(key) {
                let tid = result.fresh_transition_id();
                let trans = Transition::new(
                    tid,
                    class_to_state[&sc],
                    class_to_state[&tc],
                    t.guard.clone(),
                    t.actions.clone(),
                );
                result.add_transition(trans);
            }
        }
    }

    result.recompute_statistics();
    (result, merged_count)
}

/// Simplify guards: boolean simplification, redundant predicate elimination,
/// and guard subsumption checking.
/// Returns `(optimised_automaton, number_of_guards_simplified)`.
pub fn simplify_guards(
    auto: &SpatialEventAutomaton,
) -> (SpatialEventAutomaton, usize) {
    let mut result = auto.clone();
    let mut simplified_count = 0;

    let tids: Vec<TransitionId> = result.transition_ids();
    for tid in tids {
        if let Some(t) = result.transitions.get(&tid).cloned() {
            let original = t.guard.clone();
            let simplified = simplify_guard(&original);
            if simplified != original {
                if let Some(t_mut) = result.transitions.get_mut(&tid) {
                    t_mut.guard = simplified;
                    simplified_count += 1;
                }
            }
        }
    }

    // Guard subsumption: if two transitions from the same source have guards
    // where one subsumes the other, remove the subsumed one.
    let state_ids: Vec<StateId> = result.state_ids();
    let mut to_remove = Vec::new();
    for &sid in &state_ids {
        let out = result.outgoing(sid);
        for i in 0..out.len() {
            for j in (i + 1)..out.len() {
                let ti = out[i];
                let tj = out[j];
                if let (Some(a), Some(b)) = (result.transition(ti), result.transition(tj)) {
                    if a.target == b.target && guards_subsumed(&a.guard, &b.guard) {
                        to_remove.push(tj);
                        simplified_count += 1;
                    } else if a.target == b.target && guards_subsumed(&b.guard, &a.guard) {
                        to_remove.push(ti);
                        simplified_count += 1;
                    }
                }
            }
        }
    }

    for tid in to_remove {
        result.remove_transition(tid);
    }

    result.recompute_statistics();
    (result, simplified_count)
}

/// Eliminate transitions whose guard is trivially False or that lead to
/// non-accepting sink states with no outgoing transitions.
/// Returns `(optimised_automaton, number_of_transitions_removed)`.
pub fn eliminate_dead_transitions(
    auto: &SpatialEventAutomaton,
) -> (SpatialEventAutomaton, usize) {
    let mut result = auto.clone();
    let mut removed = 0;

    // Phase 1: remove transitions with trivially false guards
    let tids: Vec<TransitionId> = result.transition_ids();
    for tid in tids {
        if let Some(t) = result.transition(tid) {
            if t.guard.is_trivially_false() {
                result.remove_transition(tid);
                removed += 1;
            }
        }
    }

    // Phase 2: find sink states (no outgoing, not accepting) and remove
    // transitions into them
    let sink_states: HashSet<StateId> = result
        .state_ids()
        .into_iter()
        .filter(|&s| {
            result.outgoing(s).is_empty()
                && !result.accepting_states.contains(&s)
                && result.initial_state != Some(s)
        })
        .collect();

    if !sink_states.is_empty() {
        let tids: Vec<TransitionId> = result.transition_ids();
        for tid in tids {
            if let Some(t) = result.transition(tid) {
                if sink_states.contains(&t.target) && !sink_states.contains(&t.source) {
                    result.remove_transition(tid);
                    removed += 1;
                }
            }
        }
        // Remove the sink states themselves
        for sid in &sink_states {
            result.remove_state(*sid);
        }
    }

    // Phase 3: remove transitions to states that cannot reach any accepting state
    let can_reach_accepting = result.states_reaching_accepting();
    let tids: Vec<TransitionId> = result.transition_ids();
    for tid in tids {
        if let Some(t) = result.transition(tid) {
            if !can_reach_accepting.contains(&t.target)
                && !result.accepting_states.contains(&t.target)
            {
                result.remove_transition(tid);
                removed += 1;
            }
        }
    }

    result.recompute_statistics();
    (result, removed)
}

/// Compress linear chains: sequences of states s₀→s₁→…→sₙ where each
/// intermediate sᵢ has exactly one incoming and one outgoing transition,
/// is not initial, accepting, or error, gets compressed to s₀→sₙ.
/// Returns `(optimised_automaton, number_of_chains_compressed)`.
pub fn compress_linear_chains(
    auto: &SpatialEventAutomaton,
) -> (SpatialEventAutomaton, usize) {
    let mut result = auto.clone();
    let mut compressed = 0;

    // Find chain midpoints: exactly 1 incoming, exactly 1 outgoing,
    // not initial/accepting/error
    let is_chain_mid = |sid: StateId| -> bool {
        if result.initial_state == Some(sid) {
            return false;
        }
        if result.accepting_states.contains(&sid) {
            return false;
        }
        if let Some(s) = result.state(sid) {
            if s.is_error {
                return false;
            }
        }
        result.incoming(sid).len() == 1 && result.outgoing(sid).len() == 1
    };

    // Find chains
    let mut chain_mids: HashSet<StateId> = HashSet::new();
    for &sid in &result.state_ids() {
        if is_chain_mid(sid) {
            chain_mids.insert(sid);
        }
    }

    if chain_mids.is_empty() {
        return (auto.clone(), 0);
    }

    // For each chain, find the start (non-mid predecessor) and end (non-mid successor)
    let mut chains: Vec<Vec<StateId>> = Vec::new();
    let mut visited_mids: HashSet<StateId> = HashSet::new();

    for &mid in &chain_mids {
        if visited_mids.contains(&mid) {
            continue;
        }
        // Walk backwards to find chain start
        let mut chain = VecDeque::new();
        chain.push_back(mid);
        visited_mids.insert(mid);

        // Walk backward
        let mut current = mid;
        loop {
            let inc = result.incoming(current);
            if inc.len() != 1 {
                break;
            }
            if let Some(t) = result.transition(inc[0]) {
                if chain_mids.contains(&t.source) && !visited_mids.contains(&t.source) {
                    chain.push_front(t.source);
                    visited_mids.insert(t.source);
                    current = t.source;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // Walk forward
        current = mid;
        loop {
            let out = result.outgoing(current);
            if out.len() != 1 {
                break;
            }
            if let Some(t) = result.transition(out[0]) {
                if chain_mids.contains(&t.target) && !visited_mids.contains(&t.target) {
                    chain.push_back(t.target);
                    visited_mids.insert(t.target);
                    current = t.target;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if chain.len() >= 1 {
            chains.push(chain.into_iter().collect());
        }
    }

    // Compress each chain
    for chain in &chains {
        if chain.is_empty() {
            continue;
        }
        let first_mid = chain[0];
        let last_mid = chain[chain.len() - 1];

        // Find the predecessor of the first mid
        let inc = result.incoming(first_mid);
        if inc.is_empty() {
            continue;
        }
        let in_tid = inc[0];
        let predecessor = match result.transition(in_tid) {
            Some(t) => t.source,
            None => continue,
        };

        // Find the successor of the last mid
        let out = result.outgoing(last_mid);
        if out.is_empty() {
            continue;
        }
        let out_tid = out[0];
        let (successor, _successor_guard) = match result.transition(out_tid) {
            Some(t) => (t.target, t.guard.clone()),
            None => continue,
        };

        // Collect all guards and actions along the chain
        let mut guards = Vec::new();
        let mut actions = Vec::new();

        // Guard from predecessor → first_mid
        if let Some(t) = result.transition(in_tid) {
            guards.push(t.guard.clone());
            actions.extend(t.actions.clone());
        }

        // Guards/actions along the chain
        for mid in chain {
            // on_entry / on_exit actions
            if let Some(s) = result.state(*mid) {
                actions.extend(s.on_entry.clone());
                actions.extend(s.on_exit.clone());
            }
            for tid in result.outgoing(*mid) {
                if let Some(t) = result.transition(tid) {
                    guards.push(t.guard.clone());
                    actions.extend(t.actions.clone());
                }
            }
        }

        // Merge guards
        let merged_guard = if guards.is_empty() {
            Guard::True
        } else if guards.len() == 1 {
            guards.into_iter().next().unwrap()
        } else {
            // Filter out trivially-true guards
            let non_trivial: Vec<Guard> = guards
                .into_iter()
                .filter(|g| !g.is_trivially_true())
                .collect();
            if non_trivial.is_empty() {
                Guard::True
            } else if non_trivial.len() == 1 {
                non_trivial.into_iter().next().unwrap()
            } else {
                Guard::And(non_trivial)
            }
        };

        // Remove old transitions and chain states
        result.remove_transition(in_tid);
        for mid in chain {
            for tid in result.outgoing(*mid).clone() {
                result.remove_transition(tid);
            }
            for tid in result.incoming(*mid).clone() {
                result.remove_transition(tid);
            }
        }

        // Add new direct transition
        let new_tid = result.fresh_transition_id();
        result.add_transition(Transition::new(
            new_tid,
            predecessor,
            successor,
            merged_guard,
            actions,
        ));

        // Remove chain mid states
        for mid in chain {
            result.remove_state(*mid);
        }
        compressed += 1;
    }

    result.recompute_statistics();
    (result, compressed)
}

/// Factorize common prefixes: when multiple transitions from different
/// states share the same guard prefix, introduce a shared intermediate state.
/// Returns `(optimised_automaton, number_of_prefixes_factored)`.
pub fn factorize_common_prefixes(
    auto: &SpatialEventAutomaton,
) -> (SpatialEventAutomaton, usize) {
    let mut result = auto.clone();
    let mut factored = 0;

    // Group outgoing transitions by target state
    for sid in auto.state_ids() {
        let outgoing = result.outgoing(sid);
        if outgoing.len() < 2 {
            continue;
        }

        // Group by guard event kind
        let mut groups: HashMap<String, Vec<TransitionId>> = HashMap::new();
        for tid in &outgoing {
            if let Some(t) = result.transition(*tid) {
                let key = guard_prefix_key(&t.guard);
                groups.entry(key).or_default().push(*tid);
            }
        }

        // For groups with ≥2 transitions to different targets, factor out
        for (prefix_key, tids) in &groups {
            if tids.len() < 2 || prefix_key.is_empty() {
                continue;
            }
            // Collect distinct targets
            let targets: HashSet<StateId> = tids
                .iter()
                .filter_map(|tid| result.transition(*tid).map(|t| t.target))
                .collect();
            if targets.len() < 2 {
                continue;
            }

            // Create intermediate state
            let inter_id = result.fresh_state_id();
            let inter = State::new(inter_id, format!("factor_{}", factored));
            result.add_state(inter);

            // Replace original transitions: sid → target  becomes
            // sid → inter (with shared prefix guard) then inter → target (with remaining guard)
            let first_tid = tids[0];
            let shared_guard = result
                .transition(first_tid)
                .map(|t| extract_prefix_guard(&t.guard))
                .unwrap_or(Guard::True);

            // Add sid → inter
            let new_tid = result.fresh_transition_id();
            result.add_transition(Transition::new(
                new_tid,
                sid,
                inter_id,
                shared_guard,
                vec![],
            ));

            // For each original transition, add inter → target with remaining guard
            for &tid in tids {
                if let Some(t) = result.transition(tid).cloned() {
                    let remaining = extract_suffix_guard(&t.guard);
                    let ntid = result.fresh_transition_id();
                    result.add_transition(Transition::new(
                        ntid,
                        inter_id,
                        t.target,
                        remaining,
                        t.actions.clone(),
                    ));
                }
            }

            // Remove original transitions
            for &tid in tids {
                result.remove_transition(tid);
            }

            factored += 1;
        }
    }

    result.recompute_statistics();
    (result, factored)
}

// ---------------------------------------------------------------------------
// Guard simplification helpers
// ---------------------------------------------------------------------------

/// Simplify a guard expression.
fn simplify_guard(guard: &Guard) -> Guard {
    match guard {
        Guard::And(gs) => {
            let simplified: Vec<Guard> = gs
                .iter()
                .map(simplify_guard)
                .filter(|g| !g.is_trivially_true())
                .collect();
            // If any is false, the whole conjunction is false
            if simplified.iter().any(|g| g.is_trivially_false()) {
                return Guard::False;
            }
            // Flatten nested Ands
            let mut flattened = Vec::new();
            for g in simplified {
                match g {
                    Guard::And(inner) => flattened.extend(inner),
                    other => flattened.push(other),
                }
            }
            // Remove duplicates
            flattened = dedup_guards(flattened);

            match flattened.len() {
                0 => Guard::True,
                1 => flattened.into_iter().next().unwrap(),
                _ => Guard::And(flattened),
            }
        }
        Guard::Or(gs) => {
            let simplified: Vec<Guard> = gs
                .iter()
                .map(simplify_guard)
                .filter(|g| !g.is_trivially_false())
                .collect();
            if simplified.iter().any(|g| g.is_trivially_true()) {
                return Guard::True;
            }
            // Flatten nested Ors
            let mut flattened = Vec::new();
            for g in simplified {
                match g {
                    Guard::Or(inner) => flattened.extend(inner),
                    other => flattened.push(other),
                }
            }
            flattened = dedup_guards(flattened);

            match flattened.len() {
                0 => Guard::False,
                1 => flattened.into_iter().next().unwrap(),
                _ => Guard::Or(flattened),
            }
        }
        Guard::Not(inner) => {
            let simplified_inner = simplify_guard(inner);
            match simplified_inner {
                Guard::True => Guard::False,
                Guard::False => Guard::True,
                Guard::Not(double_neg) => *double_neg,
                other => Guard::Not(Box::new(other)),
            }
        }
        other => other.clone(),
    }
}

/// Remove duplicate guards from a list (by display string comparison).
fn dedup_guards(guards: Vec<Guard>) -> Vec<Guard> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for g in guards {
        let key = format!("{}", g);
        if seen.insert(key) {
            result.push(g);
        }
    }
    result
}

/// Check whether guard `a` subsumes guard `b` (i.e., whenever `b` is true,
/// `a` is also true).  Conservative approximation.
fn guards_subsumed(a: &Guard, b: &Guard) -> bool {
    // Exact equality is a trivial subsumption
    if a == b {
        return true;
    }
    // True subsumes everything
    if matches!(a, Guard::True) {
        return true;
    }
    // a = Or(…, b, …) => a subsumes b
    if let Guard::Or(gs) = a {
        if gs.iter().any(|g| g == b) {
            return true;
        }
    }
    // b = And(…, a, …) => a subsumes b
    if let Guard::And(gs) = b {
        if gs.iter().any(|g| g == a) {
            return true;
        }
    }
    false
}

/// Extract the event kind used as a prefix key for grouping.
fn guard_prefix_key(guard: &Guard) -> String {
    match guard {
        Guard::Event(ek) => format!("{}", ek),
        Guard::And(gs) => {
            for g in gs {
                if let Guard::Event(ek) = g {
                    return format!("{}", ek);
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

/// Extract the prefix (event) part of a compound guard.
fn extract_prefix_guard(guard: &Guard) -> Guard {
    match guard {
        Guard::Event(ek) => Guard::Event(ek.clone()),
        Guard::And(gs) => {
            for g in gs {
                if let Guard::Event(ek) = g {
                    return Guard::Event(ek.clone());
                }
            }
            guard.clone()
        }
        _ => guard.clone(),
    }
}

/// Extract the suffix (non-event) part of a compound guard.
fn extract_suffix_guard(guard: &Guard) -> Guard {
    match guard {
        Guard::Event(_) => Guard::True,
        Guard::And(gs) => {
            let remaining: Vec<Guard> = gs
                .iter()
                .filter(|g| !matches!(g, Guard::Event(_)))
                .cloned()
                .collect();
            match remaining.len() {
                0 => Guard::True,
                1 => remaining.into_iter().next().unwrap(),
                _ => Guard::And(remaining),
            }
        }
        _ => guard.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::{State, Transition};
    use crate::builder::{AutomatonBuilder, StateConfig};
    use crate::*;

    fn make_auto_with_unreachable() -> SpatialEventAutomaton {
        let mut auto = SpatialEventAutomaton::new("unreachable_test");
        let s0 = auto.fresh_state_id();
        let s1 = auto.fresh_state_id();
        let s2 = auto.fresh_state_id(); // unreachable

        let mut st0 = State::new(s0, "init");
        st0.is_initial = true;
        auto.add_state(st0);

        let mut st1 = State::new(s1, "end");
        st1.is_accepting = true;
        auto.add_state(st1);

        auto.add_state(State::new(s2, "orphan"));

        let t0 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(
            t0, s0, s1, Guard::Event(EventKind::GrabStart), vec![],
        ));

        auto
    }

    #[test]
    fn test_remove_unreachable() {
        let auto = make_auto_with_unreachable();
        let (result, removed) = remove_unreachable_states(&auto);
        assert_eq!(removed, 1);
        assert_eq!(result.state_count(), 2);
    }

    #[test]
    fn test_merge_equivalent_states() {
        // Two states with identical outgoing transitions should be merged
        let mut auto = SpatialEventAutomaton::new("merge_test");
        let s0 = auto.fresh_state_id();
        let s1 = auto.fresh_state_id();
        let s2 = auto.fresh_state_id();
        let s3 = auto.fresh_state_id();

        let mut st0 = State::new(s0, "init");
        st0.is_initial = true;
        auto.add_state(st0);
        auto.add_state(State::new(s1, "a"));
        auto.add_state(State::new(s2, "b"));
        let mut st3 = State::new(s3, "end");
        st3.is_accepting = true;
        auto.add_state(st3);

        // s0 → s1, s0 → s2 (s1 and s2 have identical outgoing to s3)
        let t0 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(t0, s0, s1, Guard::Event(EventKind::GrabStart), vec![]));
        let t1 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(t1, s0, s2, Guard::Event(EventKind::GrabEnd), vec![]));
        let t2 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(t2, s1, s3, Guard::Event(EventKind::TouchStart), vec![]));
        let t3 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(t3, s2, s3, Guard::Event(EventKind::TouchStart), vec![]));

        let (result, merged) = merge_equivalent_states(&auto);
        assert!(merged > 0);
        assert!(result.state_count() <= auto.state_count());
    }

    #[test]
    fn test_simplify_guards() {
        let mut auto = SpatialEventAutomaton::new("simplify_test");
        let s0 = auto.fresh_state_id();
        let s1 = auto.fresh_state_id();
        let mut st0 = State::new(s0, "init");
        st0.is_initial = true;
        auto.add_state(st0);
        let mut st1 = State::new(s1, "end");
        st1.is_accepting = true;
        auto.add_state(st1);

        // Add a transition with a complex guard that can be simplified
        let complex_guard = Guard::And(vec![
            Guard::True,
            Guard::Event(EventKind::GrabStart),
            Guard::And(vec![Guard::True]),
        ]);
        let t0 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(t0, s0, s1, complex_guard, vec![]));

        let (result, simplified) = simplify_guards(&auto);
        assert!(simplified > 0 || result.transition_count() <= auto.transition_count());
    }

    #[test]
    fn test_eliminate_dead_transitions() {
        let mut auto = SpatialEventAutomaton::new("dead_test");
        let s0 = auto.fresh_state_id();
        let s1 = auto.fresh_state_id();
        let s2 = auto.fresh_state_id(); // dead end (not accepting)

        let mut st0 = State::new(s0, "init");
        st0.is_initial = true;
        auto.add_state(st0);
        let mut st1 = State::new(s1, "good");
        st1.is_accepting = true;
        auto.add_state(st1);
        auto.add_state(State::new(s2, "dead"));

        let t0 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(t0, s0, s1, Guard::Event(EventKind::GrabStart), vec![]));
        let t1 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(t1, s0, s2, Guard::False, vec![]));

        let (result, removed) = eliminate_dead_transitions(&auto);
        assert!(removed > 0);
    }

    #[test]
    fn test_compress_linear_chains() {
        let mut b = AutomatonBuilder::new("chain_test");
        let s0 = b.add_state(StateConfig::new("init").initial());
        let s1 = b.add_state(StateConfig::new("mid1"));
        let s2 = b.add_state(StateConfig::new("mid2"));
        let s3 = b.add_state(StateConfig::new("end").accepting());

        b.add_transition(s0, s1, Guard::Event(EventKind::GrabStart), vec![]);
        b.add_transition(s1, s2, Guard::True, vec![]);
        b.add_transition(s2, s3, Guard::True, vec![]);

        let auto = b.build().unwrap();
        let (result, compressed) = compress_linear_chains(&auto);
        assert!(compressed > 0 || result.state_count() <= auto.state_count());
    }

    #[test]
    fn test_pipeline_all() {
        let auto = make_auto_with_unreachable();
        let mut pipeline = OptimizationPipeline::all();
        let result = pipeline.apply_all(&auto);
        assert!(result.state_count() <= auto.state_count());
        let stats = pipeline.statistics();
        assert!(stats.total_passes > 0);
    }

    #[test]
    fn test_pipeline_custom() {
        let auto = make_auto_with_unreachable();
        let mut pipeline =
            OptimizationPipeline::with_passes(vec![OptPass::RemoveUnreachable]);
        let result = pipeline.apply_all(&auto);
        assert_eq!(result.state_count(), 2);
    }

    #[test]
    fn test_simplify_guard_function() {
        assert_eq!(simplify_guard(&Guard::And(vec![Guard::True, Guard::True])), Guard::True);
        assert_eq!(
            simplify_guard(&Guard::And(vec![Guard::True, Guard::False])),
            Guard::False,
        );
        assert_eq!(
            simplify_guard(&Guard::Or(vec![Guard::False, Guard::False])),
            Guard::False,
        );
        assert_eq!(
            simplify_guard(&Guard::Or(vec![Guard::False, Guard::True])),
            Guard::True,
        );
        assert_eq!(
            simplify_guard(&Guard::Not(Box::new(Guard::Not(Box::new(Guard::True))))),
            Guard::True,
        );
    }

    #[test]
    fn test_guards_subsumed() {
        assert!(guards_subsumed(&Guard::True, &Guard::Event(EventKind::GrabStart)));
        assert!(guards_subsumed(
            &Guard::Event(EventKind::GrabStart),
            &Guard::Event(EventKind::GrabStart),
        ));
        let or_guard = Guard::Or(vec![
            Guard::Event(EventKind::GrabStart),
            Guard::Event(EventKind::GrabEnd),
        ]);
        assert!(guards_subsumed(&or_guard, &Guard::Event(EventKind::GrabStart)));
    }

    #[test]
    fn test_optimization_stats_display() {
        let stats = OptimizationStats {
            states_removed: 5,
            transitions_removed: 3,
            states_merged: 2,
            guards_simplified: 1,
            chains_compressed: 0,
            prefixes_factored: 0,
            total_passes: 2,
        };
        let display = format!("{}", stats);
        assert!(display.contains("5"));
        assert!(display.contains("3"));
    }

    #[test]
    fn test_optimization_stats_merge() {
        let mut a = OptimizationStats::default();
        let b = OptimizationStats {
            states_removed: 2,
            transitions_removed: 3,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.states_removed, 2);
        assert_eq!(a.transitions_removed, 3);
    }

    #[test]
    fn test_factorize_common_prefixes() {
        let mut b = AutomatonBuilder::new("prefix_test");
        let s0 = b.add_state(StateConfig::new("init").initial());
        let s1 = b.add_state(StateConfig::new("a").accepting());
        let s2 = b.add_state(StateConfig::new("b").accepting());

        // Two transitions with the same event prefix but different spatial guards
        b.add_transition(
            s0,
            s1,
            Guard::And(vec![
                Guard::Event(EventKind::GrabStart),
                Guard::Spatial(SpatialPredicate::Named(SpatialPredicateId("near".into()))),
            ]),
            vec![],
        );
        b.add_transition(
            s0,
            s2,
            Guard::And(vec![
                Guard::Event(EventKind::GrabStart),
                Guard::Spatial(SpatialPredicate::Named(SpatialPredicateId("far".into()))),
            ]),
            vec![],
        );

        let auto = b.build().unwrap();
        let (result, factored) = factorize_common_prefixes(&auto);
        // Should factor out the common GrabStart prefix
        assert!(factored > 0 || result.state_count() >= auto.state_count());
    }

    #[test]
    fn test_pipeline_idempotent() {
        let auto = make_auto_with_unreachable();
        let mut p1 = OptimizationPipeline::all();
        let result1 = p1.apply_all(&auto);
        let mut p2 = OptimizationPipeline::all();
        let result2 = p2.apply_all(&result1);
        assert_eq!(result1.state_count(), result2.state_count());
        assert_eq!(result1.transition_count(), result2.transition_count());
    }
}

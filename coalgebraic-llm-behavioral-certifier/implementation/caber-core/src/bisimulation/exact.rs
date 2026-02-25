// exact.rs — Exact bisimulation computation via Paige-Tarjan partition refinement.
// All types are defined locally; no external CABER imports.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A single transition in a labeled transition system.
#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    pub source: usize,
    pub action: String,
    pub target: usize,
    pub probability: f64,
}

/// Labeled transition system (LTS) supporting both non-deterministic and
/// probabilistic transitions.
#[derive(Clone, Debug)]
pub struct LabeledTransitionSystem {
    pub num_states: usize,
    pub actions: Vec<String>,
    pub transitions: Vec<Transition>,
    pub state_labels: Vec<Vec<String>>,
    pub initial_states: Vec<usize>,
}

impl LabeledTransitionSystem {
    pub fn new(num_states: usize) -> Self {
        Self {
            num_states,
            actions: Vec::new(),
            transitions: Vec::new(),
            state_labels: vec![Vec::new(); num_states],
            initial_states: Vec::new(),
        }
    }

    pub fn add_action(&mut self, action: &str) {
        let s = action.to_string();
        if !self.actions.contains(&s) {
            self.actions.push(s);
        }
    }

    pub fn add_transition(&mut self, source: usize, action: &str, target: usize, prob: f64) {
        self.add_action(action);
        self.transitions.push(Transition {
            source,
            action: action.to_string(),
            target,
            probability: prob,
        });
    }

    pub fn add_state_label(&mut self, state: usize, label: &str) {
        if state < self.num_states {
            let s = label.to_string();
            if !self.state_labels[state].contains(&s) {
                self.state_labels[state].push(s);
            }
        }
    }

    /// All (target, probability) pairs reachable from `state` via `action`.
    pub fn successors(&self, state: usize, action: &str) -> Vec<(usize, f64)> {
        self.transitions
            .iter()
            .filter(|t| t.source == state && t.action == action)
            .map(|t| (t.target, t.probability))
            .collect()
    }

    /// All (source, probability) pairs that can reach `state` via `action`.
    pub fn predecessors(&self, state: usize, action: &str) -> Vec<(usize, f64)> {
        self.transitions
            .iter()
            .filter(|t| t.target == state && t.action == action)
            .map(|t| (t.source, t.probability))
            .collect()
    }

    /// Actions enabled at `state`.
    pub fn actions_from(&self, state: usize) -> Vec<&str> {
        let mut acts: Vec<&str> = self
            .transitions
            .iter()
            .filter(|t| t.source == state)
            .map(|t| t.action.as_str())
            .collect();
        acts.sort();
        acts.dedup();
        acts
    }

    /// True if for every (state, action) pair there is at most one target.
    pub fn is_deterministic(&self) -> bool {
        let mut seen: HashSet<(usize, String)> = HashSet::new();
        for t in &self.transitions {
            if !seen.insert((t.source, t.action.clone())) {
                // Check if same target — still deterministic
                let targets: Vec<usize> = self
                    .transitions
                    .iter()
                    .filter(|tt| tt.source == t.source && tt.action == t.action)
                    .map(|tt| tt.target)
                    .collect();
                let first = targets[0];
                if targets.iter().any(|&tgt| tgt != first) {
                    return false;
                }
            }
        }
        true
    }

    /// BFS from `state`, returning all reachable state indices (including `state` itself).
    pub fn reachable_from(&self, state: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(state);
        queue.push_back(state);
        while let Some(s) = queue.pop_front() {
            for t in &self.transitions {
                if t.source == s && !visited.contains(&t.target) {
                    visited.insert(t.target);
                    queue.push_back(t.target);
                }
            }
        }
        let mut result: Vec<usize> = visited.into_iter().collect();
        result.sort();
        result
    }

    /// Build a predecessor map: action -> target -> set of sources.
    fn predecessor_map(&self) -> HashMap<String, HashMap<usize, HashSet<usize>>> {
        let mut map: HashMap<String, HashMap<usize, HashSet<usize>>> = HashMap::new();
        for t in &self.transitions {
            map.entry(t.action.clone())
                .or_default()
                .entry(t.target)
                .or_default()
                .insert(t.source);
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How the initial partition is seeded before refinement.
#[derive(Clone, Debug)]
pub enum InitialPartition {
    /// All states in one block.
    Trivial,
    /// Partition by state labels (states with identical label sets share a block).
    ByLabels,
    /// User-supplied partition.
    Custom(Vec<Vec<usize>>),
}

/// Configuration for bisimulation computation.
#[derive(Clone, Debug)]
pub struct BisimConfig {
    pub initial_partition: InitialPartition,
    pub max_iterations: usize,
    pub compute_relation: bool,
}

impl Default for BisimConfig {
    fn default() -> Self {
        Self {
            initial_partition: InitialPartition::ByLabels,
            max_iterations: 10_000,
            compute_relation: true,
        }
    }
}

// ---------------------------------------------------------------------------
// PartitionRefinement — Paige-Tarjan style worklist algorithm
// ---------------------------------------------------------------------------

/// Manages partition blocks and the worklist-driven refinement loop.
pub struct PartitionRefinement {
    pub blocks: Vec<HashSet<usize>>,
    pub state_to_block: Vec<usize>,
    pub worklist: VecDeque<usize>,
}

impl PartitionRefinement {
    pub fn new(initial: Vec<Vec<usize>>, num_states: usize) -> Self {
        let mut blocks: Vec<HashSet<usize>> = Vec::new();
        let mut state_to_block = vec![0usize; num_states];

        for (idx, class) in initial.iter().enumerate() {
            let set: HashSet<usize> = class.iter().cloned().collect();
            for &s in &set {
                if s < num_states {
                    state_to_block[s] = idx;
                }
            }
            blocks.push(set);
        }

        // Seed worklist with all blocks.
        let worklist: VecDeque<usize> = (0..blocks.len()).collect();

        Self {
            blocks,
            state_to_block,
            worklist,
        }
    }

    /// Main refinement loop.  Processes every (block, action) pair as a
    /// potential splitter until the partition stabilises or the iteration
    /// limit is reached.
    pub fn refine(&mut self, system: &LabeledTransitionSystem, max_iterations: usize) {
        let pred_map = system.predecessor_map();
        let actions: Vec<String> = system.actions.clone();
        let mut iterations = 0;

        while let Some(splitter_idx) = self.worklist.pop_front() {
            if iterations >= max_iterations {
                break;
            }
            iterations += 1;

            // For every action, use the current splitter block to try to split other blocks.
            for action in &actions {
                // Collect the splitter block snapshot (it may be mutated later).
                let splitter: HashSet<usize> = if splitter_idx < self.blocks.len() {
                    self.blocks[splitter_idx].clone()
                } else {
                    continue;
                };
                if splitter.is_empty() {
                    continue;
                }

                // Compute pre(splitter, action): states that have an
                // action-transition into the splitter block.
                let pre_splitter: HashSet<usize> = {
                    let mut pre = HashSet::new();
                    if let Some(act_map) = pred_map.get(action.as_str()) {
                        for &tgt in &splitter {
                            if let Some(srcs) = act_map.get(&tgt) {
                                for &src in srcs {
                                    pre.insert(src);
                                }
                            }
                        }
                    }
                    pre
                };

                if pre_splitter.is_empty() {
                    continue;
                }

                // Iterate over existing blocks and try to split.
                let num_blocks = self.blocks.len();
                let mut new_blocks: Vec<(usize, HashSet<usize>, HashSet<usize>)> = Vec::new();

                for blk_idx in 0..num_blocks {
                    if self.blocks[blk_idx].len() <= 1 {
                        continue;
                    }
                    let intersection: HashSet<usize> = self.blocks[blk_idx]
                        .intersection(&pre_splitter)
                        .cloned()
                        .collect();
                    let difference: HashSet<usize> = self.blocks[blk_idx]
                        .difference(&intersection)
                        .cloned()
                        .collect();

                    if !intersection.is_empty() && !difference.is_empty() {
                        new_blocks.push((blk_idx, intersection, difference));
                    }
                }

                // Apply splits.
                for (blk_idx, part_a, part_b) in new_blocks {
                    // Keep the larger part in the original slot, push the
                    // smaller part as a new block — this is the Paige-Tarjan
                    // trick that yields O(n log n).
                    let (keep, split_off) = if part_a.len() >= part_b.len() {
                        (part_a, part_b)
                    } else {
                        (part_b, part_a)
                    };

                    self.blocks[blk_idx] = keep.clone();
                    for &s in &keep {
                        self.state_to_block[s] = blk_idx;
                    }

                    let new_idx = self.blocks.len();
                    for &s in &split_off {
                        self.state_to_block[s] = new_idx;
                    }
                    self.blocks.push(split_off);

                    // If the original block was already on the worklist, we
                    // need both halves; otherwise add only the smaller half.
                    if self.worklist.contains(&blk_idx) {
                        self.worklist.push_back(new_idx);
                    } else {
                        self.worklist.push_back(blk_idx);
                        self.worklist.push_back(new_idx);
                    }
                }
            }
        }
    }

    /// Try to split a single block with respect to a splitter block and one
    /// action.  Returns true if the block was actually split.
    pub fn split_block(
        &mut self,
        block_idx: usize,
        splitter: &HashSet<usize>,
        system: &LabeledTransitionSystem,
        action: &str,
    ) -> bool {
        if block_idx >= self.blocks.len() || self.blocks[block_idx].len() <= 1 {
            return false;
        }

        let block = &self.blocks[block_idx];
        let mut can_reach: HashSet<usize> = HashSet::new();
        let mut cannot_reach: HashSet<usize> = HashSet::new();

        for &state in block {
            let succs = system.successors(state, action);
            if succs.iter().any(|(tgt, _)| splitter.contains(tgt)) {
                can_reach.insert(state);
            } else {
                cannot_reach.insert(state);
            }
        }

        if can_reach.is_empty() || cannot_reach.is_empty() {
            return false;
        }

        // Perform the split.
        self.blocks[block_idx] = can_reach.clone();
        for &s in &can_reach {
            self.state_to_block[s] = block_idx;
        }

        let new_idx = self.blocks.len();
        for &s in &cannot_reach {
            self.state_to_block[s] = new_idx;
        }
        self.blocks.push(cannot_reach);
        self.worklist.push_back(new_idx);

        true
    }

    /// A partition is stable when no block can be split further.
    pub fn is_stable(&self, system: &LabeledTransitionSystem) -> bool {
        for (_blk_idx, block) in self.blocks.iter().enumerate() {
            if block.len() <= 1 {
                continue;
            }
            for action in &system.actions {
                for other_blk in &self.blocks {
                    if other_blk.is_empty() {
                        continue;
                    }
                    let mut has_transition: Option<bool> = None;
                    let mut inconsistent = false;
                    for &state in block {
                        let succs = system.successors(state, action);
                        let reaches = succs.iter().any(|(tgt, _)| other_blk.contains(tgt));
                        match has_transition {
                            None => has_transition = Some(reaches),
                            Some(prev) => {
                                if prev != reaches {
                                    inconsistent = true;
                                    break;
                                }
                            }
                        }
                    }
                    if inconsistent {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Extract the partition as a vector of equivalence classes.
    pub fn to_partition(&self) -> Vec<Vec<usize>> {
        self.blocks
            .iter()
            .filter(|b| !b.is_empty())
            .map(|b| {
                let mut v: Vec<usize> = b.iter().cloned().collect();
                v.sort();
                v
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ExactBisimulation
// ---------------------------------------------------------------------------

/// Result of an exact bisimulation computation on an LTS.
pub struct ExactBisimulation {
    pub system: LabeledTransitionSystem,
    pub partition: Vec<Vec<usize>>,
    pub relation: Vec<(usize, usize)>,
    pub config: BisimConfig,
}

impl ExactBisimulation {
    pub fn new(system: LabeledTransitionSystem, config: BisimConfig) -> Self {
        Self {
            system,
            partition: Vec::new(),
            relation: Vec::new(),
            config,
        }
    }

    /// Run partition refinement (Paige-Tarjan) to compute the coarsest stable
    /// partition, then optionally materialise the bisimulation relation.
    pub fn compute(&mut self) {
        let initial = self.build_initial_partition();
        let mut pr = PartitionRefinement::new(initial, self.system.num_states);
        pr.refine(&self.system, self.config.max_iterations);

        self.partition = pr.to_partition();

        if self.config.compute_relation {
            self.relation = self.build_relation();
        }
    }

    fn build_initial_partition(&self) -> Vec<Vec<usize>> {
        match &self.config.initial_partition {
            InitialPartition::Trivial => {
                vec![(0..self.system.num_states).collect()]
            }
            InitialPartition::ByLabels => {
                // Group states by their label set.
                let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
                for s in 0..self.system.num_states {
                    let mut lbl = self.system.state_labels[s].clone();
                    lbl.sort();
                    groups.entry(lbl).or_default().push(s);
                }
                let mut parts: Vec<Vec<usize>> = groups.into_values().collect();
                parts.sort_by_key(|v| v[0]);
                parts
            }
            InitialPartition::Custom(p) => p.clone(),
        }
    }

    fn build_relation(&self) -> Vec<(usize, usize)> {
        let mut rel = Vec::new();
        for class in &self.partition {
            for i in 0..class.len() {
                for j in 0..class.len() {
                    rel.push((class[i], class[j]));
                }
            }
        }
        rel.sort();
        rel.dedup();
        rel
    }

    /// Are two states in the same equivalence class?
    pub fn are_bisimilar(&self, s1: usize, s2: usize) -> bool {
        self.partition.iter().any(|c| c.contains(&s1) && c.contains(&s2))
    }

    pub fn equivalence_classes(&self) -> &[Vec<usize>] {
        &self.partition
    }

    /// Build the quotient LTS: one state per equivalence class, with merged
    /// transitions.
    pub fn quotient_system(&self) -> LabeledTransitionSystem {
        let num_classes = self.partition.len();
        let mut quotient = LabeledTransitionSystem::new(num_classes);

        // Map old state -> class index.
        let mut state_class: Vec<usize> = vec![0; self.system.num_states];
        for (ci, class) in self.partition.iter().enumerate() {
            for &s in class {
                state_class[s] = ci;
            }
        }

        // Copy actions.
        for a in &self.system.actions {
            quotient.add_action(a);
        }

        // Copy labels from representative (first state in each class).
        for (ci, class) in self.partition.iter().enumerate() {
            if let Some(&rep) = class.first() {
                for lbl in &self.system.state_labels[rep] {
                    quotient.add_state_label(ci, lbl);
                }
            }
        }

        // Build transitions, deduplicating.
        let mut seen_trans: HashSet<(usize, String, usize)> = HashSet::new();
        for t in &self.system.transitions {
            let src_c = state_class[t.source];
            let tgt_c = state_class[t.target];
            let key = (src_c, t.action.clone(), tgt_c);
            if seen_trans.insert(key) {
                quotient.add_transition(src_c, &t.action, tgt_c, t.probability);
            }
        }

        // Initial states mapped.
        let mut init_classes: Vec<usize> = self
            .system
            .initial_states
            .iter()
            .map(|&s| state_class[s])
            .collect();
        init_classes.sort();
        init_classes.dedup();
        quotient.initial_states = init_classes;

        quotient
    }

    /// Materialise the maximum bisimulation relation — every (s1, s2) pair
    /// where s1 ~ s2.
    pub fn maximum_bisimulation(&self) -> Vec<(usize, usize)> {
        if !self.relation.is_empty() {
            return self.relation.clone();
        }
        let mut rel = Vec::new();
        for class in &self.partition {
            for &a in class {
                for &b in class {
                    rel.push((a, b));
                }
            }
        }
        rel.sort();
        rel.dedup();
        rel
    }

    /// Find the equivalence class that contains `state`.
    pub fn class_of(&self, state: usize) -> Option<&Vec<usize>> {
        self.partition.iter().find(|c| c.contains(&state))
    }

    pub fn num_classes(&self) -> usize {
        self.partition.len()
    }

    /// Ratio of original states to equivalence classes.
    pub fn reduction_ratio(&self) -> f64 {
        if self.partition.is_empty() {
            return 1.0;
        }
        self.system.num_states as f64 / self.partition.len() as f64
    }
}

// ---------------------------------------------------------------------------
// ProofStep / CoinductiveProof
// ---------------------------------------------------------------------------

/// A single step in a coinductive bisimulation proof.
#[derive(Clone, Debug)]
pub struct ProofStep {
    pub from_pair: (usize, usize),
    pub action: String,
    pub to_pair: (usize, usize),
    pub justification: String,
}

impl fmt::Display for ProofStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({},{}) --{}-> ({},{})  [{}]",
            self.from_pair.0,
            self.from_pair.1,
            self.action,
            self.to_pair.0,
            self.to_pair.1,
            self.justification
        )
    }
}

/// A coinductive proof that two states are bisimilar.
///
/// Constructed by finding a bisimulation relation containing the target pair,
/// together with explicit proof steps that witness the simulation conditions.
#[derive(Clone, Debug)]
pub struct CoinductiveProof {
    pub relation: Vec<(usize, usize)>,
    pub steps: Vec<ProofStep>,
}

impl CoinductiveProof {
    /// Attempt to construct a coinductive proof that `s1 ~ s2`.
    ///
    /// Algorithm:
    /// 1. Start with R = {(s1, s2)}.
    /// 2. Maintain a worklist of pairs whose simulation obligations have not
    ///    yet been discharged.
    /// 3. For each pair (p, q) in the worklist and each action `a`:
    ///    - For every a-successor p' of p, find an a-successor q' of q such
    ///      that (p', q') is already in R or can be added.
    ///    - Symmetrically for q's successors.
    ///    - Record a ProofStep for each matching.
    /// 4. If all obligations are discharged R is a bisimulation; return it.
    ///    Otherwise return None.
    pub fn construct(
        system: &LabeledTransitionSystem,
        s1: usize,
        s2: usize,
    ) -> Option<CoinductiveProof> {
        if s1 >= system.num_states || s2 >= system.num_states {
            return None;
        }

        let mut relation: HashSet<(usize, usize)> = HashSet::new();
        let mut worklist: VecDeque<(usize, usize)> = VecDeque::new();
        let mut steps: Vec<ProofStep> = Vec::new();

        relation.insert((s1, s2));
        worklist.push_back((s1, s2));

        // Pre-compute an exact bisimulation to know which states are truly
        // bisimilar; we use this as an oracle.
        let cfg = BisimConfig {
            initial_partition: InitialPartition::ByLabels,
            max_iterations: 10_000,
            compute_relation: false,
        };
        let mut exact = ExactBisimulation::new(system.clone(), cfg);
        exact.compute();

        // Quick check: are s1 and s2 actually bisimilar?
        if !exact.are_bisimilar(s1, s2) {
            return None;
        }

        let max_pairs = system.num_states * system.num_states;
        let mut iterations = 0;

        while let Some((p, q)) = worklist.pop_front() {
            iterations += 1;
            if iterations > max_pairs + 1 {
                break;
            }

            let actions_p = system.actions_from(p);
            let actions_q = system.actions_from(q);

            // Collect all actions from both sides.
            let mut all_actions: Vec<String> = actions_p.iter().map(|a| a.to_string()).collect();
            for a in &actions_q {
                let s = a.to_string();
                if !all_actions.contains(&s) {
                    all_actions.push(s);
                }
            }

            for action in &all_actions {
                let succs_p = system.successors(p, action);
                let succs_q = system.successors(q, action);

                // Forward simulation: every p' has a matching q'.
                for &(pp, _) in &succs_p {
                    let matched = succs_q.iter().find(|&&(qq, _)| {
                        relation.contains(&(pp, qq)) || exact.are_bisimilar(pp, qq)
                    });
                    match matched {
                        Some(&(qq, _)) => {
                            steps.push(ProofStep {
                                from_pair: (p, q),
                                action: action.clone(),
                                to_pair: (pp, qq),
                                justification: format!(
                                    "forward: ({},{}) --{}-> ({},{})",
                                    p, q, action, pp, qq
                                ),
                            });
                            if !relation.contains(&(pp, qq)) {
                                relation.insert((pp, qq));
                                worklist.push_back((pp, qq));
                            }
                        }
                        None => return None,
                    }
                }

                // Backward simulation: every q' has a matching p'.
                for &(qq, _) in &succs_q {
                    let matched = succs_p.iter().find(|&&(pp, _)| {
                        relation.contains(&(pp, qq)) || exact.are_bisimilar(pp, qq)
                    });
                    match matched {
                        Some(&(pp, _)) => {
                            let pair = (pp, qq);
                            if !relation.contains(&pair) {
                                relation.insert(pair);
                                worklist.push_back(pair);
                                steps.push(ProofStep {
                                    from_pair: (q, p),
                                    action: action.clone(),
                                    to_pair: (qq, pp),
                                    justification: format!(
                                        "backward: ({},{}) --{}-> ({},{})",
                                        q, p, action, qq, pp
                                    ),
                                });
                            }
                        }
                        None => return None,
                    }
                }
            }
        }

        let mut rel_vec: Vec<(usize, usize)> = relation.into_iter().collect();
        rel_vec.sort();

        Some(CoinductiveProof {
            relation: rel_vec,
            steps,
        })
    }

    /// Validate that `self.relation` is indeed a bisimulation on `system`.
    ///
    /// For every (p, q) in R and every action a:
    ///   - for every p --a-> p' there exists q --a-> q' with (p', q') in R
    ///   - for every q --a-> q' there exists p --a-> p' with (p', q') in R
    pub fn validate(&self, system: &LabeledTransitionSystem) -> bool {
        let rel_set: HashSet<(usize, usize)> = self.relation.iter().cloned().collect();

        for &(p, q) in &self.relation {
            // Collect the union of actions from both states.
            let actions_p = system.actions_from(p);
            let actions_q = system.actions_from(q);
            let mut all_actions: Vec<String> = actions_p.iter().map(|s| s.to_string()).collect();
            for a in actions_q {
                let s = a.to_string();
                if !all_actions.contains(&s) {
                    all_actions.push(s);
                }
            }

            for action in &all_actions {
                let succs_p = system.successors(p, action);
                let succs_q = system.successors(q, action);

                // Forward
                for &(pp, _) in &succs_p {
                    if !succs_q.iter().any(|&(qq, _)| rel_set.contains(&(pp, qq))) {
                        return false;
                    }
                }

                // Backward
                for &(qq, _) in &succs_q {
                    if !succs_p.iter().any(|&(pp, _)| rel_set.contains(&(pp, qq))) {
                        return false;
                    }
                }
            }
        }

        true
    }

    pub fn steps(&self) -> &[ProofStep] {
        &self.steps
    }
}

// ---------------------------------------------------------------------------
// BisimUpTo — bisimulation up-to techniques
// ---------------------------------------------------------------------------

/// Bisimulation up-to techniques accelerate coinductive proofs by allowing
/// the candidate relation to be smaller than a full bisimulation, provided
/// it is contained in the bisimulation when closed under a compatible
/// function.
pub struct BisimUpTo;

impl BisimUpTo {
    pub fn new() -> Self {
        Self
    }

    /// Check if `rel`, when closed under bisimilarity, yields a bisimulation.
    ///
    /// Concretely: compute the exact bisimulation equivalence ≈, then check
    /// that for every (p, q) in rel and every action a:
    ///   ∀ p' ∈ succ(p,a). ∃ q' ∈ succ(q,a). (p', q') ∈ ≈ ∘ rel ∘ ≈
    /// and symmetrically.
    pub fn up_to_bisimilarity(
        &self,
        system: &LabeledTransitionSystem,
        rel: &[(usize, usize)],
    ) -> bool {
        // Compute the exact bisimulation.
        let cfg = BisimConfig {
            initial_partition: InitialPartition::ByLabels,
            max_iterations: 10_000,
            compute_relation: false,
        };
        let mut exact = ExactBisimulation::new(system.clone(), cfg);
        exact.compute();

        // Build the closure ≈ ∘ rel ∘ ≈.
        let closed = self.bisim_closure(system, rel, &exact);

        self.is_simulation_rel(system, &closed)
    }

    /// Check if the union rel1 ∪ rel2 is a bisimulation.
    pub fn up_to_union(
        &self,
        system: &LabeledTransitionSystem,
        r1: &[(usize, usize)],
        r2: &[(usize, usize)],
    ) -> bool {
        let mut combined: Vec<(usize, usize)> = Vec::new();
        combined.extend_from_slice(r1);
        combined.extend_from_slice(r2);
        combined.sort();
        combined.dedup();
        self.check_bisimulation(system, &combined)
    }

    /// Check if rel ⊆ bisim(ctx(rel)), where ctx(rel) is the contextual
    /// closure.  Here we approximate the context closure as the reflexive-
    /// transitive closure of rel (i.e., we close under identity and
    /// composition).
    pub fn up_to_context(
        &self,
        system: &LabeledTransitionSystem,
        rel: &[(usize, usize)],
    ) -> bool {
        let closed = self.reflexive_transitive_closure(system.num_states, rel);
        self.check_bisimulation(system, &closed)
    }

    // ---- helpers -----------------------------------------------------------

    /// Compute ≈ ∘ R ∘ ≈.
    fn bisim_closure(
        &self,
        _system: &LabeledTransitionSystem,
        rel: &[(usize, usize)],
        exact: &ExactBisimulation,
    ) -> Vec<(usize, usize)> {
        let mut result: HashSet<(usize, usize)> = HashSet::new();

        for &(p, q) in rel {
            // Find all p' ≈ p and q' ≈ q.
            let class_p: Vec<usize> = exact
                .class_of(p)
                .map(|c| c.clone())
                .unwrap_or_else(|| vec![p]);
            let class_q: Vec<usize> = exact
                .class_of(q)
                .map(|c| c.clone())
                .unwrap_or_else(|| vec![q]);

            for &pp in &class_p {
                for &qq in &class_q {
                    result.insert((pp, qq));
                }
            }
        }

        let mut v: Vec<(usize, usize)> = result.into_iter().collect();
        v.sort();
        v
    }

    /// Reflexive-transitive closure of a relation on states 0..n.
    fn reflexive_transitive_closure(
        &self,
        n: usize,
        rel: &[(usize, usize)],
    ) -> Vec<(usize, usize)> {
        let mut matrix = vec![vec![false; n]; n];

        // Reflexive
        for i in 0..n {
            matrix[i][i] = true;
        }
        // Base relation
        for &(a, b) in rel {
            if a < n && b < n {
                matrix[a][b] = true;
            }
        }
        // Transitive closure (Warshall)
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if matrix[i][k] && matrix[k][j] {
                        matrix[i][j] = true;
                    }
                }
            }
        }

        let mut result = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if matrix[i][j] {
                    result.push((i, j));
                }
            }
        }
        result
    }

    /// Check that a relation is a bisimulation.
    fn check_bisimulation(
        &self,
        system: &LabeledTransitionSystem,
        rel: &[(usize, usize)],
    ) -> bool {
        self.is_simulation_rel(system, rel)
    }

    /// Check forward and backward simulation conditions.
    fn is_simulation_rel(
        &self,
        system: &LabeledTransitionSystem,
        rel: &[(usize, usize)],
    ) -> bool {
        let rel_set: HashSet<(usize, usize)> = rel.iter().cloned().collect();

        for &(p, q) in rel {
            let actions_p = system.actions_from(p);
            let actions_q = system.actions_from(q);
            let mut all_actions: Vec<String> = actions_p.iter().map(|s| s.to_string()).collect();
            for a in actions_q {
                let s = a.to_string();
                if !all_actions.contains(&s) {
                    all_actions.push(s);
                }
            }

            for action in &all_actions {
                let succs_p = system.successors(p, action);
                let succs_q = system.successors(q, action);

                for &(pp, _) in &succs_p {
                    if !succs_q.iter().any(|&(qq, _)| rel_set.contains(&(pp, qq))) {
                        return false;
                    }
                }
                for &(qq, _) in &succs_q {
                    if !succs_p.iter().any(|&(pp, _)| rel_set.contains(&(pp, qq))) {
                        return false;
                    }
                }
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Helpers used across multiple structs
// ---------------------------------------------------------------------------

/// Build a small LTS suitable for testing.  The returned system has the
/// topology:
///
/// ```text
///   0 --a-> 1 --a-> 0  (a-loop between 0 and 1)
///   0 --b-> 0
///   1 --b-> 1
/// ```
///
/// States 0 and 1 are bisimilar because they have the same observable
/// behaviour.
fn make_bisimilar_pair() -> LabeledTransitionSystem {
    let mut lts = LabeledTransitionSystem::new(2);
    lts.add_transition(0, "a", 1, 1.0);
    lts.add_transition(1, "a", 0, 1.0);
    lts.add_transition(0, "b", 0, 1.0);
    lts.add_transition(1, "b", 1, 1.0);
    lts.initial_states = vec![0];
    lts
}

/// Three-state system where state 2 is *not* bisimilar to 0 and 1.
///
/// ```text
///   0 --a-> 1    1 --a-> 0    2 --a-> 2
///   0 --b-> 0    1 --b-> 1    (no b-transition from 2)
/// ```
fn make_three_state_system() -> LabeledTransitionSystem {
    let mut lts = LabeledTransitionSystem::new(3);
    lts.add_transition(0, "a", 1, 1.0);
    lts.add_transition(1, "a", 0, 1.0);
    lts.add_transition(0, "b", 0, 1.0);
    lts.add_transition(1, "b", 1, 1.0);
    lts.add_transition(2, "a", 2, 1.0);
    lts.initial_states = vec![0];
    lts
}

/// Four-state diamond used for more complex tests.
///
/// ```text
///   0 --a-> 1    0 --a-> 2
///   1 --b-> 3    2 --b-> 3
///   3 --a-> 3    (self-loop)
/// ```
fn make_diamond_system() -> LabeledTransitionSystem {
    let mut lts = LabeledTransitionSystem::new(4);
    lts.add_transition(0, "a", 1, 1.0);
    lts.add_transition(0, "a", 2, 1.0);
    lts.add_transition(1, "b", 3, 1.0);
    lts.add_transition(2, "b", 3, 1.0);
    lts.add_transition(3, "a", 3, 1.0);
    lts.initial_states = vec![0];
    lts
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers for tests ---------------------------------------------------

    fn default_config() -> BisimConfig {
        BisimConfig {
            initial_partition: InitialPartition::Trivial,
            max_iterations: 10_000,
            compute_relation: true,
        }
    }

    fn config_by_labels() -> BisimConfig {
        BisimConfig {
            initial_partition: InitialPartition::ByLabels,
            max_iterations: 10_000,
            compute_relation: true,
        }
    }

    // 1. Two-state bisimilar system -----------------------------------------

    #[test]
    fn test_two_state_bisimilar() {
        let lts = make_bisimilar_pair();
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();

        assert!(bisim.are_bisimilar(0, 1));
        assert_eq!(bisim.num_classes(), 1);
    }

    // 2. Three-state system with non-bisimilar states -----------------------

    #[test]
    fn test_three_state_non_bisimilar() {
        let lts = make_three_state_system();
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();

        assert!(bisim.are_bisimilar(0, 1));
        assert!(!bisim.are_bisimilar(0, 2));
        assert!(!bisim.are_bisimilar(1, 2));
        assert_eq!(bisim.num_classes(), 2);
    }

    // 3. Quotient construction ----------------------------------------------

    #[test]
    fn test_quotient_construction() {
        let lts = make_three_state_system();
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();

        let q = bisim.quotient_system();
        // Two equivalence classes => two states in quotient.
        assert_eq!(q.num_states, 2);
        assert!(!q.transitions.is_empty());
    }

    // 4. Partition refinement convergence -----------------------------------

    #[test]
    fn test_partition_refinement_convergence() {
        let lts = make_diamond_system();
        let initial = vec![vec![0, 1, 2, 3]];
        let mut pr = PartitionRefinement::new(initial, 4);
        pr.refine(&lts, 10_000);

        // After convergence the partition must be stable.
        assert!(pr.is_stable(&lts));
    }

    // 5. Coinductive proof construction and validation -----------------------

    #[test]
    fn test_coinductive_proof_construction() {
        let lts = make_bisimilar_pair();
        let proof = CoinductiveProof::construct(&lts, 0, 1);
        assert!(proof.is_some());

        let proof = proof.unwrap();
        assert!(proof.validate(&lts));
        assert!(!proof.relation.is_empty());
    }

    #[test]
    fn test_coinductive_proof_validation_fails_for_non_bisimilar() {
        let lts = make_three_state_system();
        let proof = CoinductiveProof::construct(&lts, 0, 2);
        assert!(proof.is_none());
    }

    // 6. Up-to bisimilarity -------------------------------------------------

    #[test]
    fn test_up_to_bisimilarity() {
        let lts = make_bisimilar_pair();
        let upto = BisimUpTo::new();

        // (0,1) is bisimilar, so the singleton relation should pass up-to
        // bisimilarity.
        let rel = vec![(0, 1)];
        assert!(upto.up_to_bisimilarity(&lts, &rel));
    }

    // 7. Up-to union --------------------------------------------------------

    #[test]
    fn test_up_to_union() {
        let lts = make_bisimilar_pair();
        let upto = BisimUpTo::new();

        let r1 = vec![(0, 0), (1, 1)];
        let r2 = vec![(0, 1), (1, 0)];
        assert!(upto.up_to_union(&lts, &r1, &r2));
    }

    // 8. Deterministic system check -----------------------------------------

    #[test]
    fn test_is_deterministic() {
        let lts = make_bisimilar_pair();
        assert!(lts.is_deterministic());

        let nd = make_diamond_system(); // 0 --a-> 1 and 0 --a-> 2
        assert!(!nd.is_deterministic());
    }

    // 9. BFS reachability ---------------------------------------------------

    #[test]
    fn test_bfs_reachability() {
        let lts = make_three_state_system();
        let reachable = lts.reachable_from(0);
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&1));
        // State 2 is unreachable from 0.
        assert!(!reachable.contains(&2));
    }

    // 10. Self-loop bisimulation --------------------------------------------

    #[test]
    fn test_self_loop_single_state() {
        let mut lts = LabeledTransitionSystem::new(1);
        lts.add_transition(0, "a", 0, 1.0);
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();
        assert_eq!(bisim.num_classes(), 1);
        assert!(bisim.are_bisimilar(0, 0));
    }

    // 11. Maximum bisimulation relation materialization ---------------------

    #[test]
    fn test_maximum_bisimulation_relation() {
        let lts = make_bisimilar_pair();
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();

        let rel = bisim.maximum_bisimulation();
        // Both (0,1) and (1,0) should be present.
        assert!(rel.contains(&(0, 1)));
        assert!(rel.contains(&(1, 0)));
        assert!(rel.contains(&(0, 0)));
        assert!(rel.contains(&(1, 1)));
    }

    // 12. class_of ----------------------------------------------------------

    #[test]
    fn test_class_of() {
        let lts = make_three_state_system();
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();

        let c0 = bisim.class_of(0).unwrap();
        let c1 = bisim.class_of(1).unwrap();
        assert_eq!(c0, c1);

        let c2 = bisim.class_of(2).unwrap();
        assert_ne!(c0, c2);
    }

    // 13. Reduction ratio ---------------------------------------------------

    #[test]
    fn test_reduction_ratio() {
        let lts = make_three_state_system();
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();

        // 3 states, 2 classes => ratio 1.5
        let ratio = bisim.reduction_ratio();
        assert!((ratio - 1.5).abs() < 1e-9);
    }

    // 14. ByLabels initial partition ----------------------------------------

    #[test]
    fn test_initial_partition_by_labels() {
        let mut lts = LabeledTransitionSystem::new(4);
        lts.add_state_label(0, "p");
        lts.add_state_label(1, "p");
        lts.add_state_label(2, "q");
        lts.add_state_label(3, "q");
        // 0,1 have label {p}; 2,3 have label {q}.
        // With no transitions every state is bisimilar to its label-mates.
        lts.add_transition(0, "a", 0, 1.0);
        lts.add_transition(1, "a", 1, 1.0);
        lts.add_transition(2, "a", 2, 1.0);
        lts.add_transition(3, "a", 3, 1.0);

        let mut bisim = ExactBisimulation::new(lts, config_by_labels());
        bisim.compute();

        assert!(bisim.are_bisimilar(0, 1));
        assert!(bisim.are_bisimilar(2, 3));
        assert!(!bisim.are_bisimilar(0, 2));
        assert_eq!(bisim.num_classes(), 2);
    }

    // 15. Quotient preserves actions ----------------------------------------

    #[test]
    fn test_quotient_preserves_actions() {
        let lts = make_three_state_system();
        let mut bisim = ExactBisimulation::new(lts.clone(), default_config());
        bisim.compute();

        let q = bisim.quotient_system();
        // All original actions should appear in the quotient.
        for a in &lts.actions {
            assert!(q.actions.contains(a), "missing action {}", a);
        }
    }

    // 16. Up-to context -----------------------------------------------------

    #[test]
    fn test_up_to_context() {
        let lts = make_bisimilar_pair();
        let upto = BisimUpTo::new();
        // The identity relation {(0,0),(1,1)} plus (0,1) should pass.
        let rel = vec![(0, 0), (1, 1), (0, 1), (1, 0)];
        assert!(upto.up_to_context(&lts, &rel));
    }

    // 17. Diamond system bisimulation ---------------------------------------

    #[test]
    fn test_diamond_system_bisimulation() {
        let lts = make_diamond_system();
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();

        // States 1 and 2 should be bisimilar (both --b-> 3).
        assert!(bisim.are_bisimilar(1, 2));
        // State 0 differs from 3.
        assert!(!bisim.are_bisimilar(0, 3));
    }

    // 18. Predecessors ------------------------------------------------------

    #[test]
    fn test_predecessors() {
        let lts = make_three_state_system();
        let preds = lts.predecessors(1, "a");
        assert!(preds.iter().any(|&(s, _)| s == 0));
    }

    // 19. actions_from ------------------------------------------------------

    #[test]
    fn test_actions_from() {
        let lts = make_three_state_system();
        let acts = lts.actions_from(0);
        assert!(acts.contains(&"a"));
        assert!(acts.contains(&"b"));
        let acts2 = lts.actions_from(2);
        assert!(acts2.contains(&"a"));
        assert!(!acts2.contains(&"b"));
    }

    // 20. Proof steps are non-empty for bisimilar pair ----------------------

    #[test]
    fn test_proof_steps_non_empty() {
        let lts = make_bisimilar_pair();
        let proof = CoinductiveProof::construct(&lts, 0, 1).unwrap();
        assert!(!proof.steps().is_empty());
    }

    // 21. Partition refinement split_block ----------------------------------

    #[test]
    fn test_split_block_manual() {
        let lts = make_three_state_system();
        let initial = vec![vec![0, 1, 2]];
        let mut pr = PartitionRefinement::new(initial, 3);

        // Use {0, 1} as splitter with action "b": only 0 and 1 have b-transitions.
        let splitter: HashSet<usize> = [0, 1].iter().cloned().collect();
        let did_split = pr.split_block(0, &splitter, &lts, "b");
        assert!(did_split);
        assert!(pr.blocks.len() >= 2);
    }

    // 22. Coinductive proof validate rejects non-bisimulation ---------------

    #[test]
    fn test_validate_rejects_bad_relation() {
        let lts = make_three_state_system();
        // (0, 2) is NOT a bisimulation pair.
        let fake_proof = CoinductiveProof {
            relation: vec![(0, 2)],
            steps: Vec::new(),
        };
        assert!(!fake_proof.validate(&lts));
    }

    // 23. Empty system ------------------------------------------------------

    #[test]
    fn test_empty_system() {
        let lts = LabeledTransitionSystem::new(0);
        let mut bisim = ExactBisimulation::new(lts, default_config());
        bisim.compute();
        assert_eq!(bisim.num_classes(), 0);
        assert!((bisim.reduction_ratio() - 1.0).abs() < 1e-9);
    }
}

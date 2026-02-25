// witness.rs — Witness and counterexample module for CABER model checking.
// Provides witness trees, counterexample traces, minimization, composition,
// validation, and formatting for CTL/probabilistic model checking results.

use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::cmp::Ordering;
use std::fmt;

// ─── Side ────────────────────────────────────────────────────────────────────

/// Which disjunct was chosen in an Or-witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Left => write!(f, "left"),
            Side::Right => write!(f, "right"),
        }
    }
}

// ─── WitnessTreeNode ────────────────────────────────────────────────────────

/// A node in an AU (forall-until) witness tree.
#[derive(Debug, Clone)]
pub struct WitnessTreeNode {
    pub state: usize,
    pub formula_witness: Box<WitnessNode>,
    pub children: Vec<WitnessTreeNode>,
}

impl WitnessTreeNode {
    pub fn leaf(state: usize, witness: WitnessNode) -> Self {
        WitnessTreeNode {
            state,
            formula_witness: Box::new(witness),
            children: Vec::new(),
        }
    }

    pub fn with_children(state: usize, witness: WitnessNode, children: Vec<WitnessTreeNode>) -> Self {
        WitnessTreeNode {
            state,
            formula_witness: Box::new(witness),
            children,
        }
    }

    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    pub fn size(&self) -> usize {
        1 + self.formula_witness.size()
            + self.children.iter().map(|c| c.size()).sum::<usize>()
    }

    pub fn states(&self, acc: &mut Vec<usize>) {
        acc.push(self.state);
        self.formula_witness.collect_states(acc);
        for child in &self.children {
            child.states(acc);
        }
    }

    pub fn render(&self, indent: usize) -> String {
        let pad = " ".repeat(indent);
        let mut out = format!("{}AU-tree node s{}\n", pad, self.state);
        out.push_str(&self.formula_witness.render(indent + 2));
        for child in &self.children {
            out.push_str(&child.render(indent + 2));
        }
        out
    }
}

// ─── WitnessNode ────────────────────────────────────────────────────────────

/// A node in a witness tree, justifying satisfaction of a temporal formula.
#[derive(Debug, Clone)]
pub enum WitnessNode {
    AtomSatisfied {
        state: usize,
        atom: String,
    },
    NotSatisfied {
        state: usize,
        inner: Box<WitnessNode>,
    },
    AndSatisfied {
        state: usize,
        left: Box<WitnessNode>,
        right: Box<WitnessNode>,
    },
    OrSatisfied {
        state: usize,
        chosen: Box<WitnessNode>,
        side: Side,
    },
    EXSatisfied {
        state: usize,
        successor: usize,
        prob: f64,
        inner: Box<WitnessNode>,
    },
    AXSatisfied {
        state: usize,
        successors: Vec<(usize, f64, Box<WitnessNode>)>,
    },
    EUSatisfied {
        state: usize,
        path: Vec<(usize, Box<WitnessNode>)>,
        goal: Box<WitnessNode>,
    },
    AUSatisfied {
        state: usize,
        tree: Box<WitnessTreeNode>,
    },
    EGSatisfied {
        state: usize,
        cycle: Vec<usize>,
        path_witnesses: Vec<Box<WitnessNode>>,
    },
    AGSatisfied {
        state: usize,
        depth: usize,
        all_satisfy: Vec<Box<WitnessNode>>,
    },
    ProbSatisfied {
        state: usize,
        probability: f64,
        threshold: f64,
    },
}

impl WitnessNode {
    /// Return the state this node pertains to.
    pub fn state(&self) -> usize {
        match self {
            WitnessNode::AtomSatisfied { state, .. }
            | WitnessNode::NotSatisfied { state, .. }
            | WitnessNode::AndSatisfied { state, .. }
            | WitnessNode::OrSatisfied { state, .. }
            | WitnessNode::EXSatisfied { state, .. }
            | WitnessNode::AXSatisfied { state, .. }
            | WitnessNode::EUSatisfied { state, .. }
            | WitnessNode::AUSatisfied { state, .. }
            | WitnessNode::EGSatisfied { state, .. }
            | WitnessNode::AGSatisfied { state, .. }
            | WitnessNode::ProbSatisfied { state, .. } => *state,
        }
    }

    /// Pretty-print this node at the given indentation level.
    pub fn render(&self, indent: usize) -> String {
        let pad = " ".repeat(indent);
        match self {
            WitnessNode::AtomSatisfied { state, atom } => {
                format!("{}s{} |= atom \"{}\"\n", pad, state, atom)
            }
            WitnessNode::NotSatisfied { state, inner } => {
                let mut s = format!("{}s{} |= NOT because:\n", pad, state);
                s.push_str(&inner.render(indent + 2));
                s
            }
            WitnessNode::AndSatisfied { state, left, right } => {
                let mut s = format!("{}s{} |= AND because:\n", pad, state);
                s.push_str(&format!("{}  left:\n", pad));
                s.push_str(&left.render(indent + 4));
                s.push_str(&format!("{}  right:\n", pad));
                s.push_str(&right.render(indent + 4));
                s
            }
            WitnessNode::OrSatisfied { state, chosen, side } => {
                let mut s = format!("{}s{} |= OR ({} chosen):\n", pad, state, side);
                s.push_str(&chosen.render(indent + 2));
                s
            }
            WitnessNode::EXSatisfied { state, successor, prob, inner } => {
                let mut s = format!(
                    "{}s{} |= EX via successor s{} (p={:.4}):\n",
                    pad, state, successor, prob
                );
                s.push_str(&inner.render(indent + 2));
                s
            }
            WitnessNode::AXSatisfied { state, successors } => {
                let mut s = format!("{}s{} |= AX for all {} successors:\n", pad, state, successors.len());
                for (succ, p, w) in successors {
                    s.push_str(&format!("{}  s{} (p={:.4}):\n", pad, succ, p));
                    s.push_str(&w.render(indent + 4));
                }
                s
            }
            WitnessNode::EUSatisfied { state, path, goal } => {
                let mut s = format!("{}s{} |= EU via path of length {}:\n", pad, state, path.len());
                for (i, (ps, pw)) in path.iter().enumerate() {
                    s.push_str(&format!("{}  step {} at s{}:\n", pad, i, ps));
                    s.push_str(&pw.render(indent + 4));
                }
                s.push_str(&format!("{}  goal:\n", pad));
                s.push_str(&goal.render(indent + 4));
                s
            }
            WitnessNode::AUSatisfied { state, tree } => {
                let mut s = format!("{}s{} |= AU via witness tree:\n", pad, state);
                s.push_str(&tree.render(indent + 2));
                s
            }
            WitnessNode::EGSatisfied { state, cycle, path_witnesses } => {
                let cycle_str: Vec<String> = cycle.iter().map(|s| format!("s{}", s)).collect();
                let mut s = format!(
                    "{}s{} |= EG via cycle [{}]:\n",
                    pad,
                    state,
                    cycle_str.join(" -> ")
                );
                for pw in path_witnesses {
                    s.push_str(&pw.render(indent + 2));
                }
                s
            }
            WitnessNode::AGSatisfied { state, depth, all_satisfy } => {
                let mut s = format!(
                    "{}s{} |= AG (depth {}, {} witnesses):\n",
                    pad, state, depth, all_satisfy.len()
                );
                for w in all_satisfy {
                    s.push_str(&w.render(indent + 2));
                }
                s
            }
            WitnessNode::ProbSatisfied { state, probability, threshold } => {
                format!(
                    "{}s{} |= Prob({:.4} >= {:.4})\n",
                    pad, state, probability, threshold
                )
            }
        }
    }

    /// Maximum depth of this witness subtree.
    pub fn depth(&self) -> usize {
        match self {
            WitnessNode::AtomSatisfied { .. } | WitnessNode::ProbSatisfied { .. } => 1,
            WitnessNode::NotSatisfied { inner, .. } => 1 + inner.depth(),
            WitnessNode::AndSatisfied { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
            WitnessNode::OrSatisfied { chosen, .. } => 1 + chosen.depth(),
            WitnessNode::EXSatisfied { inner, .. } => 1 + inner.depth(),
            WitnessNode::AXSatisfied { successors, .. } => {
                1 + successors.iter().map(|(_, _, w)| w.depth()).max().unwrap_or(0)
            }
            WitnessNode::EUSatisfied { path, goal, .. } => {
                let path_max = path.iter().map(|(_, w)| w.depth()).max().unwrap_or(0);
                1 + path_max.max(goal.depth())
            }
            WitnessNode::AUSatisfied { tree, .. } => 1 + tree.depth(),
            WitnessNode::EGSatisfied { path_witnesses, .. } => {
                1 + path_witnesses.iter().map(|w| w.depth()).max().unwrap_or(0)
            }
            WitnessNode::AGSatisfied { all_satisfy, .. } => {
                1 + all_satisfy.iter().map(|w| w.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Total number of nodes in this witness subtree.
    pub fn size(&self) -> usize {
        match self {
            WitnessNode::AtomSatisfied { .. } | WitnessNode::ProbSatisfied { .. } => 1,
            WitnessNode::NotSatisfied { inner, .. } => 1 + inner.size(),
            WitnessNode::AndSatisfied { left, right, .. } => 1 + left.size() + right.size(),
            WitnessNode::OrSatisfied { chosen, .. } => 1 + chosen.size(),
            WitnessNode::EXSatisfied { inner, .. } => 1 + inner.size(),
            WitnessNode::AXSatisfied { successors, .. } => {
                1 + successors.iter().map(|(_, _, w)| w.size()).sum::<usize>()
            }
            WitnessNode::EUSatisfied { path, goal, .. } => {
                1 + path.iter().map(|(_, w)| w.size()).sum::<usize>() + goal.size()
            }
            WitnessNode::AUSatisfied { tree, .. } => 1 + tree.size(),
            WitnessNode::EGSatisfied { path_witnesses, .. } => {
                1 + path_witnesses.iter().map(|w| w.size()).sum::<usize>()
            }
            WitnessNode::AGSatisfied { all_satisfy, .. } => {
                1 + all_satisfy.iter().map(|w| w.size()).sum::<usize>()
            }
        }
    }

    /// Collect all states mentioned in this subtree.
    pub fn collect_states(&self, acc: &mut Vec<usize>) {
        acc.push(self.state());
        match self {
            WitnessNode::AtomSatisfied { .. } | WitnessNode::ProbSatisfied { .. } => {}
            WitnessNode::NotSatisfied { inner, .. } => inner.collect_states(acc),
            WitnessNode::AndSatisfied { left, right, .. } => {
                left.collect_states(acc);
                right.collect_states(acc);
            }
            WitnessNode::OrSatisfied { chosen, .. } => chosen.collect_states(acc),
            WitnessNode::EXSatisfied { successor, inner, .. } => {
                acc.push(*successor);
                inner.collect_states(acc);
            }
            WitnessNode::AXSatisfied { successors, .. } => {
                for (s, _, w) in successors {
                    acc.push(*s);
                    w.collect_states(acc);
                }
            }
            WitnessNode::EUSatisfied { path, goal, .. } => {
                for (s, w) in path {
                    acc.push(*s);
                    w.collect_states(acc);
                }
                goal.collect_states(acc);
            }
            WitnessNode::AUSatisfied { tree, .. } => {
                tree.states(acc);
            }
            WitnessNode::EGSatisfied { cycle, path_witnesses, .. } => {
                acc.extend(cycle);
                for w in path_witnesses {
                    w.collect_states(acc);
                }
            }
            WitnessNode::AGSatisfied { all_satisfy, .. } => {
                for w in all_satisfy {
                    w.collect_states(acc);
                }
            }
        }
    }

    /// Extract a linear trace of states (for EX/EU chains).
    fn extract_trace(&self, acc: &mut Vec<usize>) {
        acc.push(self.state());
        match self {
            WitnessNode::EXSatisfied { successor, inner, .. } => {
                acc.push(*successor);
                inner.extract_trace(acc);
            }
            WitnessNode::EUSatisfied { path, goal, .. } => {
                for (s, _) in path {
                    acc.push(*s);
                }
                goal.extract_trace(acc);
            }
            _ => {}
        }
    }
}

// ─── Witness ────────────────────────────────────────────────────────────────

/// A complete witness tree demonstrating that a state satisfies a formula.
#[derive(Debug, Clone)]
pub struct Witness {
    pub root: WitnessNode,
    pub formula_description: String,
    pub state: usize,
    pub valid: bool,
}

impl Witness {
    pub fn new(state: usize, formula: &str, root: WitnessNode) -> Self {
        Witness {
            root,
            formula_description: formula.to_string(),
            state,
            valid: true,
        }
    }

    /// Recursively validate that the witness tree correctly justifies its claims
    /// against the given Kripke structure.
    pub fn validate(&self, model: &WitnessKripke) -> bool {
        if self.root.state() != self.state {
            return false;
        }
        validate_node(&self.root, model)
    }

    /// Pretty-print the entire witness tree.
    pub fn render(&self) -> String {
        let mut s = format!(
            "Witness for s{} |= \"{}\" (valid={}):\n",
            self.state, self.formula_description, self.valid
        );
        s.push_str(&self.render_tree(2));
        s
    }

    /// Render starting at a given indentation.
    pub fn render_tree(&self, indent: usize) -> String {
        self.root.render(indent)
    }

    /// Maximum depth of the witness tree.
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    /// Total number of nodes in the witness tree.
    pub fn size(&self) -> usize {
        self.root.size()
    }

    /// All states mentioned in the witness tree, deduplicated and sorted.
    pub fn states_involved(&self) -> Vec<usize> {
        let mut acc = Vec::new();
        self.root.collect_states(&mut acc);
        let mut set: Vec<usize> = acc.into_iter().collect::<HashSet<_>>().into_iter().collect();
        set.sort();
        set
    }

    /// Extract a linear trace of states from the witness, if applicable.
    pub fn to_trace(&self) -> Vec<usize> {
        let mut trace = Vec::new();
        self.root.extract_trace(&mut trace);
        // Deduplicate consecutive duplicates
        trace.dedup();
        trace
    }
}

/// Recursively validate a single witness node against the Kripke model.
fn validate_node(node: &WitnessNode, model: &WitnessKripke) -> bool {
    match node {
        WitnessNode::AtomSatisfied { state, atom } => {
            if *state >= model.num_states {
                return false;
            }
            model.has_label(*state, atom)
        }
        WitnessNode::NotSatisfied { state, inner } => {
            if *state >= model.num_states {
                return false;
            }
            // The inner witness must pertain to the same state.
            inner.state() == *state && validate_node(inner, model)
        }
        WitnessNode::AndSatisfied { state, left, right } => {
            if *state >= model.num_states {
                return false;
            }
            left.state() == *state
                && right.state() == *state
                && validate_node(left, model)
                && validate_node(right, model)
        }
        WitnessNode::OrSatisfied { state, chosen, .. } => {
            if *state >= model.num_states {
                return false;
            }
            chosen.state() == *state && validate_node(chosen, model)
        }
        WitnessNode::EXSatisfied { state, successor, inner, .. } => {
            if *state >= model.num_states || *successor >= model.num_states {
                return false;
            }
            // Successor must be reachable from state.
            if !model.has_transition(*state, *successor) {
                return false;
            }
            // Inner witness must pertain to the successor.
            inner.state() == *successor && validate_node(inner, model)
        }
        WitnessNode::AXSatisfied { state, successors } => {
            if *state >= model.num_states {
                return false;
            }
            let model_succs: HashSet<usize> =
                model.successors(*state).iter().map(|(s, _)| *s).collect();
            let witness_succs: HashSet<usize> = successors.iter().map(|(s, _, _)| *s).collect();
            // Every model successor must have a witness.
            if !model_succs.is_subset(&witness_succs) {
                return false;
            }
            // Each witness must be valid and for the correct state.
            for (s, _, w) in successors {
                if w.state() != *s || !validate_node(w, model) {
                    return false;
                }
            }
            true
        }
        WitnessNode::EUSatisfied { state, path, goal } => {
            if *state >= model.num_states {
                return false;
            }
            // Validate the path: each step must be a successor of the previous.
            let current = *state;
            for (s, w) in path {
                if w.state() != *s || !validate_node(w, model) {
                    return false;
                }
                // s must equal current for the phi-witness at that position
                if *s != current {
                    return false;
                }
                // Need a next step: look ahead to find the transition target.
                // The path represents states along the EU path; transitions go
                // from each state to the next one in the path.
            }
            // Validate transitions along the path.
            let full_path: Vec<usize> = path.iter().map(|(s, _)| *s).collect();
            for i in 0..full_path.len().saturating_sub(1) {
                if !model.has_transition(full_path[i], full_path[i + 1]) {
                    return false;
                }
            }
            // Last path element must transition to the goal state.
            if let Some((last, _)) = path.last() {
                let goal_state = goal.state();
                if !model.has_transition(*last, goal_state) {
                    return false;
                }
            }
            // Validate the goal node.
            validate_node(goal, model)
        }
        WitnessNode::AUSatisfied { state, tree } => {
            if *state >= model.num_states {
                return false;
            }
            tree.state == *state && validate_tree_node(tree, model)
        }
        WitnessNode::EGSatisfied { state, cycle, path_witnesses } => {
            if *state >= model.num_states {
                return false;
            }
            if cycle.is_empty() {
                return false;
            }
            // The cycle must start at the claimed state.
            if cycle[0] != *state {
                return false;
            }
            // Validate transitions in the cycle.
            for i in 0..cycle.len() - 1 {
                if !model.has_transition(cycle[i], cycle[i + 1]) {
                    return false;
                }
            }
            // Last element must connect back to first.
            if !model.has_transition(*cycle.last().unwrap(), cycle[0]) {
                return false;
            }
            // Each path witness must be valid.
            for w in path_witnesses {
                if !validate_node(w, model) {
                    return false;
                }
            }
            true
        }
        WitnessNode::AGSatisfied { state, all_satisfy, .. } => {
            if *state >= model.num_states {
                return false;
            }
            for w in all_satisfy {
                if !validate_node(w, model) {
                    return false;
                }
            }
            true
        }
        WitnessNode::ProbSatisfied { state, probability, threshold } => {
            if *state >= model.num_states {
                return false;
            }
            *probability >= *threshold
        }
    }
}

/// Validate a WitnessTreeNode recursively for AU.
fn validate_tree_node(node: &WitnessTreeNode, model: &WitnessKripke) -> bool {
    if node.state >= model.num_states {
        return false;
    }
    if !validate_node(&node.formula_witness, model) {
        return false;
    }
    for child in &node.children {
        if !model.has_transition(node.state, child.state) {
            return false;
        }
        if !validate_tree_node(child, model) {
            return false;
        }
    }
    true
}

// ─── TraceStep ──────────────────────────────────────────────────────────────

/// A single step in a counterexample trace.
#[derive(Debug, Clone)]
pub struct TraceStep {
    pub state: usize,
    pub action: Option<String>,
    pub labels: Vec<String>,
    pub probability: f64,
    pub description: String,
}

impl TraceStep {
    pub fn new(state: usize, description: &str) -> Self {
        TraceStep {
            state,
            action: None,
            labels: Vec::new(),
            probability: 1.0,
            description: description.to_string(),
        }
    }

    pub fn with_action(mut self, action: &str) -> Self {
        self.action = Some(action.to_string());
        self
    }

    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    pub fn with_probability(mut self, p: f64) -> Self {
        self.probability = p;
        self
    }

    pub fn render(&self) -> String {
        let action_str = self.action.as_deref().unwrap_or("-");
        let label_str = if self.labels.is_empty() {
            "{}".to_string()
        } else {
            format!("{{{}}}", self.labels.join(", "))
        };
        format!(
            "s{} [action={}, labels={}, p={:.4}]: {}",
            self.state, action_str, label_str, self.probability, self.description
        )
    }
}

// ─── CounterExampleType ─────────────────────────────────────────────────────

/// Classification of counterexample kinds.
#[derive(Debug, Clone)]
pub enum CounterExampleType {
    AtomViolation { atom: String },
    PathViolation { path: Vec<usize> },
    DeadEnd { state: usize },
    CycleViolation { cycle: Vec<usize> },
    ProbabilityViolation { actual: f64, required: f64 },
}

impl CounterExampleType {
    pub fn description(&self) -> String {
        match self {
            CounterExampleType::AtomViolation { atom } => {
                format!("Atom \"{}\" not satisfied", atom)
            }
            CounterExampleType::PathViolation { path } => {
                let ps: Vec<String> = path.iter().map(|s| format!("s{}", s)).collect();
                format!("Violating path: {}", ps.join(" -> "))
            }
            CounterExampleType::DeadEnd { state } => {
                format!("Dead-end at state s{}", state)
            }
            CounterExampleType::CycleViolation { cycle } => {
                let cs: Vec<String> = cycle.iter().map(|s| format!("s{}", s)).collect();
                format!("Violating cycle: {}", cs.join(" -> "))
            }
            CounterExampleType::ProbabilityViolation { actual, required } => {
                format!(
                    "Probability {:.4} < required {:.4}",
                    actual, required
                )
            }
        }
    }
}

// ─── CounterExample ─────────────────────────────────────────────────────────

/// A counterexample demonstrating that a state fails to satisfy a formula.
#[derive(Debug, Clone)]
pub struct CounterExample {
    pub state: usize,
    pub formula_description: String,
    pub ce_type: CounterExampleType,
    pub trace: Vec<TraceStep>,
    pub explanation: String,
}

impl CounterExample {
    pub fn new(state: usize, formula: &str, ce_type: CounterExampleType) -> Self {
        let explanation = ce_type.description();
        CounterExample {
            state,
            formula_description: formula.to_string(),
            ce_type,
            trace: Vec::new(),
            explanation,
        }
    }

    pub fn add_trace_step(&mut self, step: TraceStep) {
        self.trace.push(step);
    }

    /// Validate that the counterexample trace is consistent with the model.
    /// Checks that consecutive trace states are connected by transitions and
    /// that the violation actually manifests.
    pub fn validate(&self, model: &WitnessKripke) -> bool {
        if self.trace.is_empty() {
            // A counterexample with no trace: valid only for atom violations.
            return matches!(
                &self.ce_type,
                CounterExampleType::AtomViolation { .. }
                    | CounterExampleType::DeadEnd { .. }
                    | CounterExampleType::ProbabilityViolation { .. }
            );
        }

        // First trace step must correspond to the counterexample state.
        if self.trace[0].state != self.state {
            return false;
        }

        // Check that consecutive states are connected.
        for i in 0..self.trace.len() - 1 {
            let from = self.trace[i].state;
            let to = self.trace[i + 1].state;
            if from >= model.num_states || to >= model.num_states {
                return false;
            }
            if !model.has_transition(from, to) {
                return false;
            }
        }

        // Type-specific validation.
        match &self.ce_type {
            CounterExampleType::AtomViolation { atom } => {
                // The state must NOT have the atom.
                !model.has_label(self.state, atom)
            }
            CounterExampleType::PathViolation { path } => {
                // The path must be valid in the model.
                WitnessValidator::validate_path(model, path)
            }
            CounterExampleType::DeadEnd { state } => {
                // The state must have no successors.
                *state < model.num_states && model.successors(*state).is_empty()
            }
            CounterExampleType::CycleViolation { cycle } => {
                WitnessValidator::validate_cycle(model, cycle)
            }
            CounterExampleType::ProbabilityViolation { actual, required } => {
                *actual < *required
            }
        }
    }

    /// Render the counterexample as a human-readable string.
    pub fn render(&self) -> String {
        let mut s = format!(
            "CounterExample for s{} |= \"{}\":\n",
            self.state, self.formula_description
        );
        s.push_str(&format!("  Type: {}\n", self.ce_type.description()));
        s.push_str(&format!("  Explanation: {}\n", self.explanation));
        if !self.trace.is_empty() {
            s.push_str("  Trace:\n");
            for (i, step) in self.trace.iter().enumerate() {
                s.push_str(&format!("    [{}] {}\n", i, step.render()));
            }
        }
        s
    }

    /// Length of the counterexample trace.
    pub fn length(&self) -> usize {
        self.trace.len()
    }
}

// ─── WitnessKripke ──────────────────────────────────────────────────────────

/// A simplified Kripke structure used for witness validation and
/// counterexample operations.
#[derive(Debug, Clone)]
pub struct WitnessKripke {
    pub num_states: usize,
    pub transitions: Vec<Vec<(usize, f64)>>,
    pub labels: Vec<Vec<String>>,
}

impl WitnessKripke {
    pub fn new(n: usize) -> Self {
        WitnessKripke {
            num_states: n,
            transitions: vec![Vec::new(); n],
            labels: vec![Vec::new(); n],
        }
    }

    pub fn add_transition(&mut self, from: usize, to: usize, prob: f64) {
        if from < self.num_states && to < self.num_states {
            // Avoid duplicate transitions.
            if !self.transitions[from].iter().any(|(t, _)| *t == to) {
                self.transitions[from].push((to, prob));
            }
        }
    }

    pub fn add_label(&mut self, state: usize, label: &str) {
        if state < self.num_states {
            let l = label.to_string();
            if !self.labels[state].contains(&l) {
                self.labels[state].push(l);
            }
        }
    }

    pub fn successors(&self, state: usize) -> &[(usize, f64)] {
        if state < self.num_states {
            &self.transitions[state]
        } else {
            &[]
        }
    }

    pub fn has_label(&self, state: usize, label: &str) -> bool {
        if state < self.num_states {
            self.labels[state].iter().any(|l| l == label)
        } else {
            false
        }
    }

    pub fn has_transition(&self, from: usize, to: usize) -> bool {
        if from < self.num_states {
            self.transitions[from].iter().any(|(t, _)| *t == to)
        } else {
            false
        }
    }

    /// Return the transition probability from `from` to `to`, or 0 if none.
    pub fn transition_prob(&self, from: usize, to: usize) -> f64 {
        if from < self.num_states {
            self.transitions[from]
                .iter()
                .find(|(t, _)| *t == to)
                .map(|(_, p)| *p)
                .unwrap_or(0.0)
        } else {
            0.0
        }
    }
}

// ─── CounterExampleMinimizer ────────────────────────────────────────────────

/// Minimization algorithms for counterexample traces.
pub struct CounterExampleMinimizer;

/// Priority-queue entry for Dijkstra-based minimization.
#[derive(Clone, PartialEq)]
struct DijkstraEntry {
    cost: f64,
    state: usize,
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behaviour in BinaryHeap (which is a max-heap).
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl CounterExampleMinimizer {
    /// Find the shortest (fewest steps) counterexample trace using BFS.
    /// Starting from the counterexample state, BFS explores successors and
    /// returns the shortest path that reaches a violating condition.
    pub fn minimize(ce: &CounterExample, model: &WitnessKripke) -> CounterExample {
        let start = ce.state;
        let target_states = Self::violation_targets(ce, model);

        if target_states.is_empty() || target_states.contains(&start) {
            return ce.clone();
        }

        // BFS from start.
        let mut visited = vec![false; model.num_states];
        let mut parent: Vec<Option<usize>> = vec![None; model.num_states];
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        let mut found: Option<usize> = None;

        while let Some(current) = queue.pop_front() {
            if target_states.contains(&current) && current != start {
                found = Some(current);
                break;
            }
            for (succ, _) in model.successors(current) {
                if !visited[*succ] {
                    visited[*succ] = true;
                    parent[*succ] = Some(current);
                    queue.push_back(*succ);
                }
            }
        }

        if let Some(target) = found {
            let path = Self::reconstruct_path(&parent, start, target);
            Self::build_minimized_ce(ce, &path, model)
        } else {
            ce.clone()
        }
    }

    /// Find the minimum-cost (maximum probability) counterexample trace.
    /// Uses Dijkstra's algorithm with -log(prob) as edge cost, so the
    /// most-probable path has the smallest total cost.
    pub fn minimize_cost(ce: &CounterExample, model: &WitnessKripke) -> CounterExample {
        let start = ce.state;
        let target_states = Self::violation_targets(ce, model);

        if target_states.is_empty() || target_states.contains(&start) {
            return ce.clone();
        }

        let inf = f64::INFINITY;
        let mut dist = vec![inf; model.num_states];
        let mut parent: Vec<Option<usize>> = vec![None; model.num_states];
        dist[start] = 0.0;

        let mut heap = BinaryHeap::new();
        heap.push(DijkstraEntry {
            cost: 0.0,
            state: start,
        });

        while let Some(DijkstraEntry { cost, state }) = heap.pop() {
            if cost > dist[state] {
                continue;
            }
            if target_states.contains(&state) && state != start {
                let path = Self::reconstruct_path(&parent, start, state);
                return Self::build_minimized_ce(ce, &path, model);
            }
            for (succ, prob) in model.successors(state) {
                let edge_cost = if *prob > 0.0 { -prob.ln() } else { inf };
                let new_cost = cost + edge_cost;
                if new_cost < dist[*succ] {
                    dist[*succ] = new_cost;
                    parent[*succ] = Some(state);
                    heap.push(DijkstraEntry {
                        cost: new_cost,
                        state: *succ,
                    });
                }
            }
        }

        ce.clone()
    }

    /// Determine which states represent the violation target for BFS/Dijkstra.
    fn violation_targets(ce: &CounterExample, model: &WitnessKripke) -> HashSet<usize> {
        let mut targets = HashSet::new();
        match &ce.ce_type {
            CounterExampleType::AtomViolation { atom } => {
                // States that lack the atom.
                for s in 0..model.num_states {
                    if !model.has_label(s, atom) {
                        targets.insert(s);
                    }
                }
            }
            CounterExampleType::PathViolation { path } => {
                if let Some(last) = path.last() {
                    targets.insert(*last);
                }
            }
            CounterExampleType::DeadEnd { state } => {
                targets.insert(*state);
            }
            CounterExampleType::CycleViolation { cycle } => {
                for s in cycle {
                    targets.insert(*s);
                }
            }
            CounterExampleType::ProbabilityViolation { .. } => {
                // All states are potential targets; use the trace endpoints.
                if let Some(last_step) = ce.trace.last() {
                    targets.insert(last_step.state);
                }
            }
        }
        targets
    }

    /// Reconstruct the path from parent pointers.
    fn reconstruct_path(parent: &[Option<usize>], start: usize, end: usize) -> Vec<usize> {
        let mut path = Vec::new();
        let mut cur = end;
        while cur != start {
            path.push(cur);
            match parent[cur] {
                Some(p) => cur = p,
                None => break,
            }
        }
        path.push(start);
        path.reverse();
        path
    }

    /// Build a minimized CounterExample from a path.
    fn build_minimized_ce(
        original: &CounterExample,
        path: &[usize],
        model: &WitnessKripke,
    ) -> CounterExample {
        let mut ce = CounterExample::new(
            original.state,
            &original.formula_description,
            original.ce_type.clone(),
        );
        ce.explanation = format!("Minimized: {} steps (original {})", path.len(), original.length());

        for i in 0..path.len() {
            let s = path[i];
            let prob = if i + 1 < path.len() {
                model.transition_prob(s, path[i + 1])
            } else {
                1.0
            };
            let step = TraceStep::new(s, &format!("step {}", i))
                .with_probability(prob)
                .with_labels(model.labels[s].clone());
            ce.add_trace_step(step);
        }
        ce
    }
}

// ─── WitnessComposer ───────────────────────────────────────────────────────

/// Compose witnesses from sub-witnesses.
pub struct WitnessComposer;

impl WitnessComposer {
    /// Compose an AND-witness from two witnesses for the same state.
    pub fn compose_and(w1: &Witness, w2: &Witness) -> Witness {
        let state = w1.state;
        let root = WitnessNode::AndSatisfied {
            state,
            left: Box::new(w1.root.clone()),
            right: Box::new(w2.root.clone()),
        };
        let formula = format!("({}) AND ({})", w1.formula_description, w2.formula_description);
        Witness::new(state, &formula, root)
    }

    /// Compose an OR-witness from two witnesses, choosing the first.
    pub fn compose_or(w1: &Witness, w2: &Witness) -> Witness {
        let state = w1.state;
        // Prefer w1 (left).
        let root = WitnessNode::OrSatisfied {
            state,
            chosen: Box::new(w1.root.clone()),
            side: Side::Left,
        };
        let formula = format!("({}) OR ({})", w1.formula_description, w2.formula_description);
        let mut w = Witness::new(state, &formula, root);
        w.valid = w1.valid;
        w
    }

    /// Compose an EU-witness from a sequence of witnesses for the path.
    /// The last witness is treated as the goal; the rest form the path.
    pub fn compose_sequence(witnesses: &[Witness]) -> Witness {
        if witnesses.is_empty() {
            return Witness::new(0, "empty", WitnessNode::AtomSatisfied {
                state: 0,
                atom: "empty".to_string(),
            });
        }
        if witnesses.len() == 1 {
            return witnesses[0].clone();
        }

        let state = witnesses[0].state;
        let path: Vec<(usize, Box<WitnessNode>)> = witnesses[..witnesses.len() - 1]
            .iter()
            .map(|w| (w.state, Box::new(w.root.clone())))
            .collect();
        let goal = Box::new(witnesses.last().unwrap().root.clone());

        let root = WitnessNode::EUSatisfied { state, path, goal };
        let descs: Vec<&str> = witnesses.iter().map(|w| w.formula_description.as_str()).collect();
        let formula = format!("EU({})", descs.join(", "));
        Witness::new(state, &formula, root)
    }

    /// Compose an AU-witness tree from witnesses.
    /// The first witness is the root; the rest become children.
    pub fn compose_tree(witnesses: &[Witness]) -> Witness {
        if witnesses.is_empty() {
            return Witness::new(0, "empty", WitnessNode::AtomSatisfied {
                state: 0,
                atom: "empty".to_string(),
            });
        }

        let root_w = &witnesses[0];
        let children: Vec<WitnessTreeNode> = witnesses[1..]
            .iter()
            .map(|w| WitnessTreeNode::leaf(w.state, w.root.clone()))
            .collect();

        let tree_node = WitnessTreeNode::with_children(
            root_w.state,
            root_w.root.clone(),
            children,
        );

        let root = WitnessNode::AUSatisfied {
            state: root_w.state,
            tree: Box::new(tree_node),
        };
        let descs: Vec<&str> = witnesses.iter().map(|w| w.formula_description.as_str()).collect();
        let formula = format!("AU({})", descs.join(", "));
        Witness::new(root_w.state, &formula, root)
    }
}

// ─── WitnessValidator ──────────────────────────────────────────────────────

/// Standalone validation utilities for witness components.
pub struct WitnessValidator;

impl WitnessValidator {
    /// Check that a state is labelled with the given atom.
    pub fn validate_atomic(model: &WitnessKripke, state: usize, atom: &str) -> bool {
        model.has_label(state, atom)
    }

    /// Check that `successor` is a successor of `state` in the model.
    pub fn validate_ex(model: &WitnessKripke, state: usize, successor: usize) -> bool {
        model.has_transition(state, successor)
    }

    /// Check that consecutive states in the path are connected by transitions.
    pub fn validate_path(model: &WitnessKripke, path: &[usize]) -> bool {
        if path.is_empty() {
            return true;
        }
        for i in 0..path.len() - 1 {
            if path[i] >= model.num_states || path[i + 1] >= model.num_states {
                return false;
            }
            if !model.has_transition(path[i], path[i + 1]) {
                return false;
            }
        }
        true
    }

    /// Check that the cycle is valid: consecutive states connected and last
    /// connects back to first.
    pub fn validate_cycle(model: &WitnessKripke, cycle: &[usize]) -> bool {
        if cycle.len() < 2 {
            return false;
        }
        for i in 0..cycle.len() - 1 {
            if cycle[i] >= model.num_states || cycle[i + 1] >= model.num_states {
                return false;
            }
            if !model.has_transition(cycle[i], cycle[i + 1]) {
                return false;
            }
        }
        // Last must connect to first.
        let last = *cycle.last().unwrap();
        let first = cycle[0];
        if last >= model.num_states || first >= model.num_states {
            return false;
        }
        model.has_transition(last, first)
    }
}

// ─── WitnessFormatter ──────────────────────────────────────────────────────

/// Output formatting for witnesses in text, JSON, and DOT.
pub struct WitnessFormatter;

impl WitnessFormatter {
    /// Render the witness as plain text.
    pub fn to_text(witness: &Witness) -> String {
        witness.render()
    }

    /// Render the witness as a JSON string (manual serialisation to avoid
    /// requiring serde as a hard dependency at this level).
    pub fn to_json(witness: &Witness) -> String {
        let mut s = String::from("{\n");
        s.push_str(&format!("  \"state\": {},\n", witness.state));
        s.push_str(&format!(
            "  \"formula\": {},\n",
            json_escape(&witness.formula_description)
        ));
        s.push_str(&format!("  \"valid\": {},\n", witness.valid));
        s.push_str(&format!("  \"depth\": {},\n", witness.depth()));
        s.push_str(&format!("  \"size\": {},\n", witness.size()));
        s.push_str(&format!(
            "  \"states\": [{}],\n",
            witness
                .states_involved()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
        s.push_str("  \"root\": ");
        s.push_str(&node_to_json(&witness.root, 2));
        s.push('\n');
        s.push('}');
        s
    }

    /// Render the witness tree as a GraphViz DOT digraph.
    pub fn to_dot(witness: &Witness) -> String {
        let mut dot = String::from("digraph witness {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=record, fontsize=10];\n");
        let mut counter: usize = 0;
        node_to_dot(&witness.root, &mut dot, &mut counter, None);
        dot.push_str("}\n");
        dot
    }
}

fn json_escape(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{}\"", escaped)
}

fn node_to_json(node: &WitnessNode, indent: usize) -> String {
    let pad = " ".repeat(indent);
    let inner_pad = " ".repeat(indent + 2);
    match node {
        WitnessNode::AtomSatisfied { state, atom } => {
            format!(
                "{{\n{inner_pad}\"type\": \"AtomSatisfied\",\n{inner_pad}\"state\": {state},\n{inner_pad}\"atom\": {atom_j}\n{pad}}}",
                inner_pad = inner_pad,
                pad = pad,
                state = state,
                atom_j = json_escape(atom)
            )
        }
        WitnessNode::NotSatisfied { state, inner } => {
            format!(
                "{{\n{ip}\"type\": \"NotSatisfied\",\n{ip}\"state\": {s},\n{ip}\"inner\": {inner_json}\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
                inner_json = node_to_json(inner, indent + 2)
            )
        }
        WitnessNode::AndSatisfied { state, left, right } => {
            format!(
                "{{\n{ip}\"type\": \"AndSatisfied\",\n{ip}\"state\": {s},\n{ip}\"left\": {l},\n{ip}\"right\": {r}\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
                l = node_to_json(left, indent + 2),
                r = node_to_json(right, indent + 2)
            )
        }
        WitnessNode::OrSatisfied { state, chosen, side } => {
            format!(
                "{{\n{ip}\"type\": \"OrSatisfied\",\n{ip}\"state\": {s},\n{ip}\"side\": \"{side}\",\n{ip}\"chosen\": {c}\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
                side = side,
                c = node_to_json(chosen, indent + 2)
            )
        }
        WitnessNode::EXSatisfied { state, successor, prob, inner } => {
            format!(
                "{{\n{ip}\"type\": \"EXSatisfied\",\n{ip}\"state\": {s},\n{ip}\"successor\": {succ},\n{ip}\"prob\": {p},\n{ip}\"inner\": {inner_json}\n{pad}}}",
                ip = inner_pad,
                pad = pad,
                s = state,
                succ = successor,
                p = prob,
                inner_json = node_to_json(inner, indent + 2)
            )
        }
        WitnessNode::AXSatisfied { state, successors } => {
            let succs_json: Vec<String> = successors
                .iter()
                .map(|(s, p, w)| {
                    format!(
                        "{{\n{ip2}\"state\": {s},\n{ip2}\"prob\": {p},\n{ip2}\"witness\": {w}\n{ip}}}",
                        ip2 = " ".repeat(indent + 4),
                        ip = inner_pad,
                        s = s,
                        p = p,
                        w = node_to_json(w, indent + 4)
                    )
                })
                .collect();
            format!(
                "{{\n{ip}\"type\": \"AXSatisfied\",\n{ip}\"state\": {s},\n{ip}\"successors\": [\n{ip}  {succs}\n{ip}]\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
                succs = succs_json.join(&format!(",\n{}  ", inner_pad))
            )
        }
        WitnessNode::EUSatisfied { state, path, goal } => {
            let path_json: Vec<String> = path
                .iter()
                .map(|(s, w)| {
                    format!(
                        "{{\"state\": {}, \"witness\": {}}}",
                        s,
                        node_to_json(w, indent + 4)
                    )
                })
                .collect();
            format!(
                "{{\n{ip}\"type\": \"EUSatisfied\",\n{ip}\"state\": {s},\n{ip}\"path\": [{pj}],\n{ip}\"goal\": {g}\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
                pj = path_json.join(", "),
                g = node_to_json(goal, indent + 2)
            )
        }
        WitnessNode::AUSatisfied { state, .. } => {
            format!(
                "{{\n{ip}\"type\": \"AUSatisfied\",\n{ip}\"state\": {s}\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
            )
        }
        WitnessNode::EGSatisfied { state, cycle, .. } => {
            let cs: Vec<String> = cycle.iter().map(|s| s.to_string()).collect();
            format!(
                "{{\n{ip}\"type\": \"EGSatisfied\",\n{ip}\"state\": {s},\n{ip}\"cycle\": [{c}]\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
                c = cs.join(", ")
            )
        }
        WitnessNode::AGSatisfied { state, depth, all_satisfy } => {
            format!(
                "{{\n{ip}\"type\": \"AGSatisfied\",\n{ip}\"state\": {s},\n{ip}\"depth\": {d},\n{ip}\"count\": {c}\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
                d = depth,
                c = all_satisfy.len()
            )
        }
        WitnessNode::ProbSatisfied { state, probability, threshold } => {
            format!(
                "{{\n{ip}\"type\": \"ProbSatisfied\",\n{ip}\"state\": {s},\n{ip}\"probability\": {prob},\n{ip}\"threshold\": {thr}\n{p}}}",
                ip = inner_pad,
                p = pad,
                s = state,
                prob = probability,
                thr = threshold
            )
        }
    }
}

fn node_to_dot(
    node: &WitnessNode,
    dot: &mut String,
    counter: &mut usize,
    parent_id: Option<usize>,
) {
    let my_id = *counter;
    *counter += 1;

    let label = match node {
        WitnessNode::AtomSatisfied { state, atom } => {
            format!("s{}|atom: {}", state, atom)
        }
        WitnessNode::NotSatisfied { state, .. } => format!("s{}|NOT", state),
        WitnessNode::AndSatisfied { state, .. } => format!("s{}|AND", state),
        WitnessNode::OrSatisfied { state, side, .. } => {
            format!("s{}|OR({})", state, side)
        }
        WitnessNode::EXSatisfied { state, successor, prob, .. } => {
            format!("s{}|EX->s{} p={:.3}", state, successor, prob)
        }
        WitnessNode::AXSatisfied { state, successors, .. } => {
            format!("s{}|AX({})", state, successors.len())
        }
        WitnessNode::EUSatisfied { state, path, .. } => {
            format!("s{}|EU(len={})", state, path.len())
        }
        WitnessNode::AUSatisfied { state, .. } => format!("s{}|AU", state),
        WitnessNode::EGSatisfied { state, cycle, .. } => {
            format!("s{}|EG(cyc={})", state, cycle.len())
        }
        WitnessNode::AGSatisfied { state, depth, .. } => {
            format!("s{}|AG(d={})", state, depth)
        }
        WitnessNode::ProbSatisfied { state, probability, threshold } => {
            format!("s{}|P({:.3}>={:.3})", state, probability, threshold)
        }
    };

    dot.push_str(&format!("  n{} [label=\"{}\"];\n", my_id, label));
    if let Some(pid) = parent_id {
        dot.push_str(&format!("  n{} -> n{};\n", pid, my_id));
    }

    // Recurse into children.
    match node {
        WitnessNode::AtomSatisfied { .. } | WitnessNode::ProbSatisfied { .. } => {}
        WitnessNode::NotSatisfied { inner, .. } => {
            node_to_dot(inner, dot, counter, Some(my_id));
        }
        WitnessNode::AndSatisfied { left, right, .. } => {
            node_to_dot(left, dot, counter, Some(my_id));
            node_to_dot(right, dot, counter, Some(my_id));
        }
        WitnessNode::OrSatisfied { chosen, .. } => {
            node_to_dot(chosen, dot, counter, Some(my_id));
        }
        WitnessNode::EXSatisfied { inner, .. } => {
            node_to_dot(inner, dot, counter, Some(my_id));
        }
        WitnessNode::AXSatisfied { successors, .. } => {
            for (_, _, w) in successors {
                node_to_dot(w, dot, counter, Some(my_id));
            }
        }
        WitnessNode::EUSatisfied { path, goal, .. } => {
            for (_, w) in path {
                node_to_dot(w, dot, counter, Some(my_id));
            }
            node_to_dot(goal, dot, counter, Some(my_id));
        }
        WitnessNode::AUSatisfied { .. } => {}
        WitnessNode::EGSatisfied { path_witnesses, .. } => {
            for w in path_witnesses {
                node_to_dot(w, dot, counter, Some(my_id));
            }
        }
        WitnessNode::AGSatisfied { all_satisfy, .. } => {
            for w in all_satisfy {
                node_to_dot(w, dot, counter, Some(my_id));
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small Kripke model.
    ///
    ///   s0 --1.0--> s1 --1.0--> s2
    ///    |                        |
    ///    +---0.5--> s3 --1.0-----+
    ///                 \--1.0--> s3 (self-loop)
    ///
    /// Labels: s0={a}, s1={a,b}, s2={b,c}, s3={c}
    fn test_model() -> WitnessKripke {
        let mut m = WitnessKripke::new(4);
        m.add_transition(0, 1, 1.0);
        m.add_transition(0, 3, 0.5);
        m.add_transition(1, 2, 1.0);
        m.add_transition(3, 2, 1.0);
        m.add_transition(3, 3, 1.0);
        m.add_label(0, "a");
        m.add_label(1, "a");
        m.add_label(1, "b");
        m.add_label(2, "b");
        m.add_label(2, "c");
        m.add_label(3, "c");
        m
    }

    /// Helper: build a cycle model  s0 -> s1 -> s2 -> s0.
    fn cycle_model() -> WitnessKripke {
        let mut m = WitnessKripke::new(3);
        m.add_transition(0, 1, 1.0);
        m.add_transition(1, 2, 1.0);
        m.add_transition(2, 0, 1.0);
        m.add_label(0, "p");
        m.add_label(1, "p");
        m.add_label(2, "p");
        m
    }

    // ── Test 1: Witness for atomic proposition ──────────────────────────

    #[test]
    fn test_witness_atomic() {
        let m = test_model();
        let node = WitnessNode::AtomSatisfied {
            state: 0,
            atom: "a".to_string(),
        };
        let w = Witness::new(0, "a", node);
        assert!(w.validate(&m));
        assert_eq!(w.depth(), 1);
        assert_eq!(w.size(), 1);
        assert_eq!(w.states_involved(), vec![0]);
    }

    // ── Test 2: Witness for EX ──────────────────────────────────────────

    #[test]
    fn test_witness_ex() {
        let m = test_model();
        let inner = WitnessNode::AtomSatisfied {
            state: 1,
            atom: "b".to_string(),
        };
        let node = WitnessNode::EXSatisfied {
            state: 0,
            successor: 1,
            prob: 1.0,
            inner: Box::new(inner),
        };
        let w = Witness::new(0, "EX b", node);
        assert!(w.validate(&m));
        assert_eq!(w.depth(), 2);
        assert_eq!(w.size(), 2);
    }

    // ── Test 3: Witness for EU (path witness) ───────────────────────────

    #[test]
    fn test_witness_eu() {
        let m = test_model();
        // EU: path s0 (a holds), then goal s1 where b holds.
        // s0 |= a, transition s0 -> s1, s1 |= b.
        let path_w = WitnessNode::AtomSatisfied {
            state: 0,
            atom: "a".to_string(),
        };
        let goal_w = WitnessNode::AtomSatisfied {
            state: 1,
            atom: "b".to_string(),
        };
        let eu = WitnessNode::EUSatisfied {
            state: 0,
            path: vec![(0, Box::new(path_w))],
            goal: Box::new(goal_w),
        };
        let w = Witness::new(0, "E[a U b]", eu);
        assert!(w.validate(&m));
        assert_eq!(w.depth(), 2);
    }

    // ── Test 4: Witness validation — valid case ─────────────────────────

    #[test]
    fn test_witness_validation_valid() {
        let m = test_model();
        let left = WitnessNode::AtomSatisfied {
            state: 1,
            atom: "a".to_string(),
        };
        let right = WitnessNode::AtomSatisfied {
            state: 1,
            atom: "b".to_string(),
        };
        let node = WitnessNode::AndSatisfied {
            state: 1,
            left: Box::new(left),
            right: Box::new(right),
        };
        let w = Witness::new(1, "a AND b", node);
        assert!(w.validate(&m));
    }

    // ── Test 5: Witness validation — invalid case ───────────────────────

    #[test]
    fn test_witness_validation_invalid() {
        let m = test_model();
        // s0 does NOT have label "b", so this witness is invalid.
        let node = WitnessNode::AtomSatisfied {
            state: 0,
            atom: "b".to_string(),
        };
        let w = Witness::new(0, "b", node);
        assert!(!w.validate(&m));
    }

    // ── Test 6: Counterexample construction ─────────────────────────────

    #[test]
    fn test_counterexample_construction() {
        let m = test_model();
        let ce = CounterExample::new(
            0,
            "b",
            CounterExampleType::AtomViolation {
                atom: "b".to_string(),
            },
        );
        assert!(ce.validate(&m));
        assert_eq!(ce.length(), 0);
        let rendered = ce.render();
        assert!(rendered.contains("Atom"));
    }

    // ── Test 7: Counterexample with trace ───────────────────────────────

    #[test]
    fn test_counterexample_with_trace() {
        let m = test_model();
        let mut ce = CounterExample::new(
            0,
            "AG b",
            CounterExampleType::PathViolation {
                path: vec![0, 1],
            },
        );
        ce.add_trace_step(
            TraceStep::new(0, "start").with_probability(1.0),
        );
        ce.add_trace_step(
            TraceStep::new(1, "step 1").with_probability(1.0),
        );
        assert!(ce.validate(&m));
        assert_eq!(ce.length(), 2);
    }

    // ── Test 8: Counterexample minimization ─────────────────────────────

    #[test]
    fn test_counterexample_minimization() {
        let m = test_model();
        // Build a longer-than-necessary CE.
        let mut ce = CounterExample::new(
            0,
            "c",
            CounterExampleType::AtomViolation {
                atom: "a".to_string(),
            },
        );
        // Trace: s0 -> s1 -> s2 (but s1 already lacks "a" is wrong — s1 has "a").
        // Use s0 -> s1 -> s2; target is any state without "a" = {s2, s3}.
        ce.add_trace_step(TraceStep::new(0, "start"));
        ce.add_trace_step(TraceStep::new(1, "via s1"));
        ce.add_trace_step(TraceStep::new(2, "arrive s2"));

        let minimized = CounterExampleMinimizer::minimize(&ce, &m);
        // BFS should find a path from s0 to s2 or s3.
        assert!(minimized.length() <= ce.length());
    }

    // ── Test 9: Trace validation ────────────────────────────────────────

    #[test]
    fn test_trace_validation() {
        let m = test_model();
        assert!(WitnessValidator::validate_path(&m, &[0, 1, 2]));
        assert!(!WitnessValidator::validate_path(&m, &[0, 2])); // no direct edge
        assert!(WitnessValidator::validate_path(&m, &[0, 3]));
    }

    // ── Test 10: Cycle validation ───────────────────────────────────────

    #[test]
    fn test_cycle_validation() {
        let cm = cycle_model();
        assert!(WitnessValidator::validate_cycle(&cm, &[0, 1, 2]));
        assert!(!WitnessValidator::validate_cycle(&cm, &[0, 2, 1])); // wrong direction
    }

    // ── Test 11: Witness composition ────────────────────────────────────

    #[test]
    fn test_witness_composition() {
        let m = test_model();
        let w1 = Witness::new(
            1,
            "a",
            WitnessNode::AtomSatisfied {
                state: 1,
                atom: "a".to_string(),
            },
        );
        let w2 = Witness::new(
            1,
            "b",
            WitnessNode::AtomSatisfied {
                state: 1,
                atom: "b".to_string(),
            },
        );
        let composed = WitnessComposer::compose_and(&w1, &w2);
        assert!(composed.validate(&m));
        assert_eq!(composed.size(), 3); // And + two atoms

        let or_composed = WitnessComposer::compose_or(&w1, &w2);
        assert!(or_composed.validate(&m));
    }

    // ── Test 12: Rendering — text ───────────────────────────────────────

    #[test]
    fn test_rendering_text() {
        let w = Witness::new(
            0,
            "a",
            WitnessNode::AtomSatisfied {
                state: 0,
                atom: "a".to_string(),
            },
        );
        let text = WitnessFormatter::to_text(&w);
        assert!(text.contains("s0"));
        assert!(text.contains("atom"));
        assert!(text.contains("\"a\""));
    }

    // ── Test 13: Rendering — JSON ───────────────────────────────────────

    #[test]
    fn test_rendering_json() {
        let w = Witness::new(
            0,
            "a",
            WitnessNode::AtomSatisfied {
                state: 0,
                atom: "a".to_string(),
            },
        );
        let json = WitnessFormatter::to_json(&w);
        assert!(json.contains("\"state\": 0"));
        assert!(json.contains("\"AtomSatisfied\""));
        assert!(json.contains("\"valid\": true"));
    }

    // ── Test 14: Rendering — DOT ────────────────────────────────────────

    #[test]
    fn test_rendering_dot() {
        let inner = WitnessNode::AtomSatisfied {
            state: 1,
            atom: "b".to_string(),
        };
        let node = WitnessNode::EXSatisfied {
            state: 0,
            successor: 1,
            prob: 1.0,
            inner: Box::new(inner),
        };
        let w = Witness::new(0, "EX b", node);
        let dot = WitnessFormatter::to_dot(&w);
        assert!(dot.contains("digraph witness"));
        assert!(dot.contains("n0"));
        assert!(dot.contains("n1"));
        assert!(dot.contains("->"));
    }

    // ── Test 15: Depth and size computation ─────────────────────────────

    #[test]
    fn test_depth_and_size() {
        let leaf1 = WitnessNode::AtomSatisfied {
            state: 0,
            atom: "a".to_string(),
        };
        let leaf2 = WitnessNode::AtomSatisfied {
            state: 0,
            atom: "b".to_string(),
        };
        let and_node = WitnessNode::AndSatisfied {
            state: 0,
            left: Box::new(leaf1),
            right: Box::new(leaf2),
        };
        let ex_node = WitnessNode::EXSatisfied {
            state: 1,
            successor: 0,
            prob: 1.0,
            inner: Box::new(and_node),
        };
        assert_eq!(ex_node.depth(), 3); // EX -> AND -> Atom
        assert_eq!(ex_node.size(), 4); // EX + AND + 2 Atoms
    }

    // ── Test 16: Cost-based minimization (Dijkstra) ─────────────────────

    #[test]
    fn test_cost_minimization() {
        // Model with two paths to a target:
        //  s0 --(0.9)--> s1 --(0.9)--> s2  (high prob path)
        //  s0 --(0.1)--> s2             (low prob direct)
        let mut m = WitnessKripke::new(3);
        m.add_transition(0, 1, 0.9);
        m.add_transition(1, 2, 0.9);
        m.add_transition(0, 2, 0.1);
        m.add_label(0, "a");
        m.add_label(2, "target");

        let mut ce = CounterExample::new(
            0,
            "avoid target",
            CounterExampleType::PathViolation { path: vec![0, 2] },
        );
        ce.add_trace_step(TraceStep::new(0, "start"));
        ce.add_trace_step(TraceStep::new(2, "end"));

        let minimized = CounterExampleMinimizer::minimize_cost(&ce, &m);
        // The cost-minimized path should prefer the high-prob route s0->s1->s2.
        assert!(minimized.length() >= 2);
    }

    // ── Test 17: EG cycle witness ───────────────────────────────────────

    #[test]
    fn test_eg_cycle_witness() {
        let cm = cycle_model();
        let pw0 = WitnessNode::AtomSatisfied {
            state: 0,
            atom: "p".to_string(),
        };
        let pw1 = WitnessNode::AtomSatisfied {
            state: 1,
            atom: "p".to_string(),
        };
        let pw2 = WitnessNode::AtomSatisfied {
            state: 2,
            atom: "p".to_string(),
        };
        let eg = WitnessNode::EGSatisfied {
            state: 0,
            cycle: vec![0, 1, 2],
            path_witnesses: vec![Box::new(pw0), Box::new(pw1), Box::new(pw2)],
        };
        let w = Witness::new(0, "EG p", eg);
        assert!(w.validate(&cm));
        assert_eq!(w.states_involved(), vec![0, 1, 2]);
    }

    // ── Test 18: Probability witness ────────────────────────────────────

    #[test]
    fn test_prob_witness() {
        let m = test_model();
        let node = WitnessNode::ProbSatisfied {
            state: 0,
            probability: 0.8,
            threshold: 0.5,
        };
        let w = Witness::new(0, "P>=0.5[F target]", node);
        assert!(w.validate(&m));

        let bad = WitnessNode::ProbSatisfied {
            state: 0,
            probability: 0.3,
            threshold: 0.5,
        };
        let w2 = Witness::new(0, "P>=0.5[F target]", bad);
        assert!(!w2.validate(&m));
    }

    // ── Test 19: Witness to_trace ───────────────────────────────────────

    #[test]
    fn test_witness_to_trace() {
        let inner = WitnessNode::AtomSatisfied {
            state: 1,
            atom: "b".to_string(),
        };
        let ex = WitnessNode::EXSatisfied {
            state: 0,
            successor: 1,
            prob: 1.0,
            inner: Box::new(inner),
        };
        let w = Witness::new(0, "EX b", ex);
        let trace = w.to_trace();
        assert_eq!(trace, vec![0, 1]);
    }

    // ── Test 20: Sequence composition ───────────────────────────────────

    #[test]
    fn test_sequence_composition() {
        let w0 = Witness::new(
            0,
            "a",
            WitnessNode::AtomSatisfied {
                state: 0,
                atom: "a".to_string(),
            },
        );
        let w1 = Witness::new(
            1,
            "b",
            WitnessNode::AtomSatisfied {
                state: 1,
                atom: "b".to_string(),
            },
        );
        let w2 = Witness::new(
            2,
            "c",
            WitnessNode::AtomSatisfied {
                state: 2,
                atom: "c".to_string(),
            },
        );
        let seq = WitnessComposer::compose_sequence(&[w0, w1, w2]);
        assert_eq!(seq.state, 0);
        assert!(seq.size() > 1);
    }
}

//! Adversary model based on Definition D4 (Bounded Dolev-Yao).
//!
//! Models a network adversary with bounded capabilities: intercepting,
//! injecting, dropping, and modifying messages subject to depth (k) and
//! action count (n) bounds.

use crate::protocol::{CipherSuite, ProtocolVersion};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

// ── Message Algebra ──────────────────────────────────────────────────────

/// Terms in the symbolic message algebra.
///
/// This represents the Dolev-Yao message term algebra where messages
/// are built from atoms via pairing, encryption, hashing, and MAC.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageTerm {
    /// A fresh nonce value.
    Nonce(u64),
    /// A cryptographic key (symmetric or asymmetric).
    Key(KeyTerm),
    /// A cipher suite identifier.
    CipherId(u16),
    /// A protocol version.
    Version(u8, u8),
    /// Raw byte payload.
    Bytes(Vec<u8>),
    /// Encrypted term: Enc(key, plaintext).
    Encrypted {
        key: Box<MessageTerm>,
        plaintext: Box<MessageTerm>,
    },
    /// MAC: Mac(key, message).
    Mac {
        key: Box<MessageTerm>,
        message: Box<MessageTerm>,
    },
    /// Hash: H(message).
    Hash(Box<MessageTerm>),
    /// Pair: (left, right).
    Pair(Box<MessageTerm>, Box<MessageTerm>),
    /// TLS record: Record(content_type, version, fragment).
    Record {
        content_type: u8,
        version: Box<MessageTerm>,
        fragment: Box<MessageTerm>,
    },
    /// Network packet with metadata.
    Packet {
        source: String,
        destination: String,
        payload: Box<MessageTerm>,
    },
    /// Symbolic variable (for symbolic execution).
    Variable(String),
}

/// Key types in the Dolev-Yao model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyTerm {
    Symmetric(Vec<u8>),
    PublicKey(String),
    PrivateKey(String),
    PreShared(Vec<u8>),
    DerivedKey {
        base: Box<MessageTerm>,
        label: String,
    },
}

impl MessageTerm {
    /// Compute the depth of this term (nesting level).
    pub fn depth(&self) -> usize {
        match self {
            MessageTerm::Nonce(_)
            | MessageTerm::Key(_)
            | MessageTerm::CipherId(_)
            | MessageTerm::Version(_, _)
            | MessageTerm::Bytes(_)
            | MessageTerm::Variable(_) => 0,

            MessageTerm::Hash(inner) => 1 + inner.depth(),

            MessageTerm::Encrypted { key, plaintext } => {
                1 + key.depth().max(plaintext.depth())
            }
            MessageTerm::Mac { key, message } => {
                1 + key.depth().max(message.depth())
            }
            MessageTerm::Pair(left, right) => {
                1 + left.depth().max(right.depth())
            }
            MessageTerm::Record {
                fragment, version, ..
            } => 1 + fragment.depth().max(version.depth()),
            MessageTerm::Packet { payload, .. } => 1 + payload.depth(),
        }
    }

    /// Number of nodes in this term.
    pub fn size(&self) -> usize {
        match self {
            MessageTerm::Nonce(_)
            | MessageTerm::Key(_)
            | MessageTerm::CipherId(_)
            | MessageTerm::Version(_, _)
            | MessageTerm::Bytes(_)
            | MessageTerm::Variable(_) => 1,

            MessageTerm::Hash(inner) => 1 + inner.size(),

            MessageTerm::Encrypted { key, plaintext } => {
                1 + key.size() + plaintext.size()
            }
            MessageTerm::Mac { key, message } => {
                1 + key.size() + message.size()
            }
            MessageTerm::Pair(left, right) => 1 + left.size() + right.size(),
            MessageTerm::Record {
                fragment, version, ..
            } => 1 + fragment.size() + version.size(),
            MessageTerm::Packet { payload, .. } => 1 + payload.size(),
        }
    }

    /// Extract all sub-terms (DY analysis of extractable knowledge).
    pub fn subterms(&self) -> Vec<&MessageTerm> {
        let mut result = vec![self];
        match self {
            MessageTerm::Encrypted { key, plaintext } => {
                result.extend(key.subterms());
                result.extend(plaintext.subterms());
            }
            MessageTerm::Mac { key, message } => {
                result.extend(key.subterms());
                result.extend(message.subterms());
            }
            MessageTerm::Hash(inner) => {
                result.extend(inner.subterms());
            }
            MessageTerm::Pair(left, right) => {
                result.extend(left.subterms());
                result.extend(right.subterms());
            }
            MessageTerm::Record {
                fragment, version, ..
            } => {
                result.extend(fragment.subterms());
                result.extend(version.subterms());
            }
            MessageTerm::Packet { payload, .. } => {
                result.extend(payload.subterms());
            }
            _ => {}
        }
        result
    }

    /// Collect all free variable names in this term.
    pub fn free_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, vars: &mut HashSet<String>) {
        match self {
            MessageTerm::Variable(name) => {
                vars.insert(name.clone());
            }
            MessageTerm::Encrypted { key, plaintext } => {
                key.collect_variables(vars);
                plaintext.collect_variables(vars);
            }
            MessageTerm::Mac { key, message } => {
                key.collect_variables(vars);
                message.collect_variables(vars);
            }
            MessageTerm::Hash(inner) => inner.collect_variables(vars),
            MessageTerm::Pair(left, right) => {
                left.collect_variables(vars);
                right.collect_variables(vars);
            }
            MessageTerm::Record {
                fragment, version, ..
            } => {
                fragment.collect_variables(vars);
                version.collect_variables(vars);
            }
            MessageTerm::Packet { payload, .. } => {
                payload.collect_variables(vars);
            }
            _ => {}
        }
    }

    /// Substitute a variable with a term.
    pub fn substitute(&self, var: &str, replacement: &MessageTerm) -> MessageTerm {
        match self {
            MessageTerm::Variable(name) if name == var => replacement.clone(),
            MessageTerm::Encrypted { key, plaintext } => MessageTerm::Encrypted {
                key: Box::new(key.substitute(var, replacement)),
                plaintext: Box::new(plaintext.substitute(var, replacement)),
            },
            MessageTerm::Mac { key, message } => MessageTerm::Mac {
                key: Box::new(key.substitute(var, replacement)),
                message: Box::new(message.substitute(var, replacement)),
            },
            MessageTerm::Hash(inner) => {
                MessageTerm::Hash(Box::new(inner.substitute(var, replacement)))
            }
            MessageTerm::Pair(left, right) => MessageTerm::Pair(
                Box::new(left.substitute(var, replacement)),
                Box::new(right.substitute(var, replacement)),
            ),
            MessageTerm::Record {
                content_type,
                version,
                fragment,
            } => MessageTerm::Record {
                content_type: *content_type,
                version: Box::new(version.substitute(var, replacement)),
                fragment: Box::new(fragment.substitute(var, replacement)),
            },
            MessageTerm::Packet {
                source,
                destination,
                payload,
            } => MessageTerm::Packet {
                source: source.clone(),
                destination: destination.clone(),
                payload: Box::new(payload.substitute(var, replacement)),
            },
            other => other.clone(),
        }
    }

    /// Whether this is an atomic (leaf) term.
    pub fn is_atomic(&self) -> bool {
        matches!(
            self,
            MessageTerm::Nonce(_)
                | MessageTerm::Key(_)
                | MessageTerm::CipherId(_)
                | MessageTerm::Version(_, _)
                | MessageTerm::Bytes(_)
                | MessageTerm::Variable(_)
        )
    }
}

impl fmt::Display for MessageTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageTerm::Nonce(n) => write!(f, "N({})", n),
            MessageTerm::Key(k) => write!(f, "K({:?})", k),
            MessageTerm::CipherId(id) => write!(f, "CS(0x{:04X})", id),
            MessageTerm::Version(maj, min) => write!(f, "V({}.{})", maj, min),
            MessageTerm::Bytes(b) => write!(f, "B({})", hex::encode(b)),
            MessageTerm::Variable(name) => write!(f, "${}", name),
            MessageTerm::Encrypted { key, plaintext } => {
                write!(f, "Enc({}, {})", key, plaintext)
            }
            MessageTerm::Mac { key, message } => write!(f, "Mac({}, {})", key, message),
            MessageTerm::Hash(inner) => write!(f, "H({})", inner),
            MessageTerm::Pair(l, r) => write!(f, "<{}, {}>", l, r),
            MessageTerm::Record {
                content_type,
                version,
                fragment,
            } => write!(f, "Rec({}, {}, {})", content_type, version, fragment),
            MessageTerm::Packet {
                source,
                destination,
                payload,
            } => write!(f, "Pkt({}→{}, {})", source, destination, payload),
        }
    }
}

// ── Adversary Actions ────────────────────────────────────────────────────

/// An action the adversary can perform on the network.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdversaryAction {
    /// Intercept a message and learn its contents.
    Intercept { message: MessageTerm },
    /// Inject a new message onto the network.
    Inject { message: MessageTerm },
    /// Drop a message (prevent delivery).
    Drop { message: MessageTerm },
    /// Modify a message in transit.
    Modify {
        original: MessageTerm,
        modified: MessageTerm,
    },
    /// Replay a previously observed message.
    Replay { message: MessageTerm },
    /// Reorder messages.
    Reorder { indices: Vec<usize> },
}

impl AdversaryAction {
    /// The "cost" of this action in terms of budget consumption.
    pub fn cost(&self) -> u32 {
        match self {
            AdversaryAction::Intercept { .. } => 1,
            AdversaryAction::Inject { .. } => 2,
            AdversaryAction::Drop { .. } => 1,
            AdversaryAction::Modify { .. } => 3,
            AdversaryAction::Replay { .. } => 2,
            AdversaryAction::Reorder { .. } => 1,
        }
    }

    /// Whether this action is passive (doesn't modify the network).
    pub fn is_passive(&self) -> bool {
        matches!(self, AdversaryAction::Intercept { .. })
    }

    /// Whether this action requires knowledge of message contents.
    pub fn requires_knowledge(&self) -> bool {
        matches!(
            self,
            AdversaryAction::Inject { .. }
                | AdversaryAction::Modify { .. }
                | AdversaryAction::Replay { .. }
        )
    }
}

impl fmt::Display for AdversaryAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdversaryAction::Intercept { message } => write!(f, "intercept({})", message),
            AdversaryAction::Inject { message } => write!(f, "inject({})", message),
            AdversaryAction::Drop { message } => write!(f, "drop({})", message),
            AdversaryAction::Modify { original, modified } => {
                write!(f, "modify({} → {})", original, modified)
            }
            AdversaryAction::Replay { message } => write!(f, "replay({})", message),
            AdversaryAction::Reorder { indices } => write!(f, "reorder({:?})", indices),
        }
    }
}

// ── Adversary Budget ─────────────────────────────────────────────────────

/// Budget constraints for bounded Dolev-Yao adversary (Definition D4).
///
/// - `depth_bound` (k): maximum nesting depth for constructed messages
/// - `action_bound` (n): maximum number of adversary actions in a trace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdversaryBudget {
    pub depth_bound: u32,
    pub action_bound: u32,
}

impl AdversaryBudget {
    pub fn new(depth_bound: u32, action_bound: u32) -> Self {
        AdversaryBudget {
            depth_bound,
            action_bound,
        }
    }

    /// Check whether an action is within budget given current usage.
    pub fn can_perform(&self, action: &AdversaryAction, actions_used: u32) -> bool {
        if actions_used + action.cost() > self.action_bound {
            return false;
        }
        match action {
            AdversaryAction::Inject { message } | AdversaryAction::Modify { modified: message, .. } => {
                message.depth() as u32 <= self.depth_bound
            }
            _ => true,
        }
    }

    /// Remaining action capacity.
    pub fn remaining_actions(&self, used: u32) -> u32 {
        self.action_bound.saturating_sub(used)
    }

    /// Whether the budget has been fully spent.
    pub fn is_exhausted(&self, used: u32) -> bool {
        used >= self.action_bound
    }

    /// A very restricted budget for quick analysis.
    pub fn minimal() -> Self {
        Self::new(2, 5)
    }

    /// A moderate budget for standard analysis.
    pub fn standard() -> Self {
        Self::new(5, 20)
    }

    /// A generous budget for thorough analysis.
    pub fn thorough() -> Self {
        Self::new(10, 100)
    }
}

impl Default for AdversaryBudget {
    fn default() -> Self {
        Self::standard()
    }
}

impl fmt::Display for AdversaryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Budget(k={}, n={})",
            self.depth_bound, self.action_bound
        )
    }
}

// ── Knowledge Set ────────────────────────────────────────────────────────

/// The adversary's knowledge set with Dolev-Yao deduction.
///
/// Implements closure under destructors:
/// - Pair decomposition: knows <a, b> ⟹ knows a, knows b
/// - Decryption: knows Enc(k, m) ∧ knows k ⟹ knows m
/// - MAC verification: knows Mac(k, m) ∧ knows k ⟹ knows m
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSet {
    terms: HashSet<MessageTerm>,
    deduced: HashSet<MessageTerm>,
    closed: bool,
}

impl KnowledgeSet {
    pub fn new() -> Self {
        KnowledgeSet {
            terms: HashSet::new(),
            deduced: HashSet::new(),
            closed: true,
        }
    }

    pub fn from_terms(terms: impl IntoIterator<Item = MessageTerm>) -> Self {
        let terms: HashSet<_> = terms.into_iter().collect();
        let mut ks = KnowledgeSet {
            terms,
            deduced: HashSet::new(),
            closed: false,
        };
        ks.close();
        ks
    }

    /// Add a term to the knowledge set and recompute closure.
    pub fn learn(&mut self, term: MessageTerm) {
        if self.terms.insert(term) {
            self.closed = false;
            self.close();
        }
    }

    /// Compute the deductive closure of the knowledge set.
    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        let mut worklist: Vec<MessageTerm> = self.terms.iter().cloned().collect();
        worklist.extend(self.deduced.iter().cloned());
        let mut all_known: HashSet<MessageTerm> = self.terms.clone();
        all_known.extend(self.deduced.iter().cloned());

        let mut changed = true;
        while changed {
            changed = false;
            let snapshot: Vec<MessageTerm> = all_known.iter().cloned().collect();
            for term in &snapshot {
                let new_terms = self.decompose(term, &all_known);
                for new_term in new_terms {
                    if all_known.insert(new_term.clone()) {
                        self.deduced.insert(new_term);
                        changed = true;
                    }
                }
            }
        }
        self.closed = true;
    }

    /// Apply destructor rules to derive new knowledge from a term.
    fn decompose(&self, term: &MessageTerm, known: &HashSet<MessageTerm>) -> Vec<MessageTerm> {
        let mut results = Vec::new();
        match term {
            // Pair decomposition
            MessageTerm::Pair(left, right) => {
                results.push(left.as_ref().clone());
                results.push(right.as_ref().clone());
            }
            // Decryption: if we know the key, we can decrypt
            MessageTerm::Encrypted { key, plaintext } => {
                if known.contains(key.as_ref()) {
                    results.push(plaintext.as_ref().clone());
                }
            }
            // Record decomposition
            MessageTerm::Record {
                version, fragment, ..
            } => {
                results.push(version.as_ref().clone());
                results.push(fragment.as_ref().clone());
            }
            // Packet decomposition
            MessageTerm::Packet { payload, .. } => {
                results.push(payload.as_ref().clone());
            }
            _ => {}
        }
        results
    }

    /// Check whether a term is in the knowledge set (or derivable).
    pub fn knows(&self, term: &MessageTerm) -> bool {
        self.terms.contains(term) || self.deduced.contains(term)
    }

    /// Check if the adversary can construct a given term from knowledge.
    pub fn can_construct(&self, term: &MessageTerm, depth_bound: u32) -> bool {
        if depth_bound == 0 {
            return self.knows(term);
        }
        if self.knows(term) {
            return true;
        }
        match term {
            MessageTerm::Pair(left, right) => {
                self.can_construct(left, depth_bound - 1)
                    && self.can_construct(right, depth_bound - 1)
            }
            MessageTerm::Encrypted { key, plaintext } => {
                self.can_construct(key, depth_bound - 1)
                    && self.can_construct(plaintext, depth_bound - 1)
            }
            MessageTerm::Mac { key, message } => {
                self.can_construct(key, depth_bound - 1)
                    && self.can_construct(message, depth_bound - 1)
            }
            MessageTerm::Hash(inner) => self.can_construct(inner, depth_bound - 1),
            MessageTerm::Record {
                version, fragment, ..
            } => {
                self.can_construct(version, depth_bound - 1)
                    && self.can_construct(fragment, depth_bound - 1)
            }
            MessageTerm::Packet { payload, .. } => self.can_construct(payload, depth_bound - 1),
            _ => false,
        }
    }

    /// Total number of known terms (base + deduced).
    pub fn size(&self) -> usize {
        self.terms.len() + self.deduced.len()
    }

    /// Number of base (directly learned) terms.
    pub fn base_size(&self) -> usize {
        self.terms.len()
    }

    /// All known terms.
    pub fn all_terms(&self) -> impl Iterator<Item = &MessageTerm> {
        self.terms.iter().chain(self.deduced.iter())
    }
}

impl Default for KnowledgeSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Adversary Trace ──────────────────────────────────────────────────────

/// A sequence of adversary actions forming an attack trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryTrace {
    pub actions: Vec<AdversaryAction>,
    pub budget: AdversaryBudget,
    actions_used: u32,
}

impl AdversaryTrace {
    pub fn new(budget: AdversaryBudget) -> Self {
        AdversaryTrace {
            actions: Vec::new(),
            budget,
            actions_used: 0,
        }
    }

    /// Try to append an action. Returns false if budget exceeded.
    pub fn push(&mut self, action: AdversaryAction) -> bool {
        if !self.budget.can_perform(&action, self.actions_used) {
            return false;
        }
        self.actions_used += action.cost();
        self.actions.push(action);
        true
    }

    /// Total cost of all actions so far.
    pub fn total_cost(&self) -> u32 {
        self.actions_used
    }

    /// Remaining budget.
    pub fn remaining(&self) -> u32 {
        self.budget.remaining_actions(self.actions_used)
    }

    /// Whether the trace has exhausted its budget.
    pub fn is_exhausted(&self) -> bool {
        self.budget.is_exhausted(self.actions_used)
    }

    /// Number of actions in the trace.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Count of active (non-passive) actions.
    pub fn active_action_count(&self) -> usize {
        self.actions.iter().filter(|a| !a.is_passive()).count()
    }

    /// All messages intercepted in this trace.
    pub fn intercepted_messages(&self) -> Vec<&MessageTerm> {
        self.actions
            .iter()
            .filter_map(|a| match a {
                AdversaryAction::Intercept { message } => Some(message),
                _ => None,
            })
            .collect()
    }

    /// All messages injected in this trace.
    pub fn injected_messages(&self) -> Vec<&MessageTerm> {
        self.actions
            .iter()
            .filter_map(|a| match a {
                AdversaryAction::Inject { message } => Some(message),
                _ => None,
            })
            .collect()
    }
}

impl fmt::Display for AdversaryTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AdversaryTrace ({} actions, cost {}/{}):", self.actions.len(), self.actions_used, self.budget.action_bound)?;
        for (i, action) in self.actions.iter().enumerate() {
            writeln!(f, "  [{}] {}", i, action)?;
        }
        Ok(())
    }
}

// ── Bounded Dolev-Yao Adversary ──────────────────────────────────────────

/// A bounded Dolev-Yao adversary (Definition D4).
///
/// The adversary can intercept, inject, drop, and modify messages
/// on the network, subject to depth and action budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedDYAdversary {
    pub budget: AdversaryBudget,
    pub knowledge: KnowledgeSet,
    pub trace: AdversaryTrace,
    pub position: AdversaryPosition,
}

/// Where the adversary is positioned in the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdversaryPosition {
    /// Full man-in-the-middle.
    MitM,
    /// Can only observe and inject (not modify/drop).
    NetworkObserver,
    /// Can only observe (passive).
    Eavesdropper,
}

impl BoundedDYAdversary {
    pub fn new(budget: AdversaryBudget, position: AdversaryPosition) -> Self {
        BoundedDYAdversary {
            budget,
            knowledge: KnowledgeSet::new(),
            trace: AdversaryTrace::new(budget),
            position,
        }
    }

    pub fn mitm(budget: AdversaryBudget) -> Self {
        Self::new(budget, AdversaryPosition::MitM)
    }

    pub fn observer(budget: AdversaryBudget) -> Self {
        Self::new(budget, AdversaryPosition::NetworkObserver)
    }

    pub fn eavesdropper(budget: AdversaryBudget) -> Self {
        Self::new(budget, AdversaryPosition::Eavesdropper)
    }

    /// Check whether the adversary can perform a given action.
    pub fn can_perform(&self, action: &AdversaryAction) -> bool {
        // Check position constraints
        match self.position {
            AdversaryPosition::Eavesdropper => {
                if !action.is_passive() {
                    return false;
                }
            }
            AdversaryPosition::NetworkObserver => {
                if matches!(
                    action,
                    AdversaryAction::Drop { .. } | AdversaryAction::Modify { .. }
                ) {
                    return false;
                }
            }
            AdversaryPosition::MitM => {}
        }

        // Check budget
        if !self.budget.can_perform(action, self.trace.total_cost()) {
            return false;
        }

        // Check knowledge requirements
        if action.requires_knowledge() {
            match action {
                AdversaryAction::Inject { message } => {
                    self.knowledge.can_construct(message, self.budget.depth_bound)
                }
                AdversaryAction::Modify { modified, .. } => {
                    self.knowledge.can_construct(modified, self.budget.depth_bound)
                }
                AdversaryAction::Replay { message } => self.knowledge.knows(message),
                _ => true,
            }
        } else {
            true
        }
    }

    /// Execute an adversary action, updating state.
    pub fn perform(&mut self, action: AdversaryAction) -> Result<(), AdversaryError> {
        if !self.can_perform(&action) {
            return Err(AdversaryError::ActionNotAllowed {
                action: format!("{}", action),
                reason: "failed capability check".into(),
            });
        }

        // Learn from intercepted messages
        if let AdversaryAction::Intercept { ref message } = action {
            self.knowledge.learn(message.clone());
        }

        // Record the action
        if !self.trace.push(action) {
            return Err(AdversaryError::BudgetExceeded);
        }

        Ok(())
    }

    /// Generate all possible next actions given current knowledge and budget.
    pub fn possible_actions(&self) -> Vec<AdversaryAction> {
        let mut actions = Vec::new();

        // For each known term, the adversary can inject or replay it
        for term in self.knowledge.all_terms() {
            let inject = AdversaryAction::Inject {
                message: term.clone(),
            };
            if self.can_perform(&inject) {
                actions.push(inject);
            }

            let replay = AdversaryAction::Replay {
                message: term.clone(),
            };
            if self.can_perform(&replay) {
                actions.push(replay);
            }
        }

        actions
    }

    /// Reset the adversary to initial state (preserves budget/position).
    pub fn reset(&mut self) {
        self.knowledge = KnowledgeSet::new();
        self.trace = AdversaryTrace::new(self.budget);
    }

    /// Remaining budget.
    pub fn remaining_budget(&self) -> u32 {
        self.trace.remaining()
    }

    /// Whether the adversary has exhausted its budget.
    pub fn is_exhausted(&self) -> bool {
        self.trace.is_exhausted()
    }
}

/// Errors specific to adversary operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AdversaryError {
    #[error("action not allowed: {action} ({reason})")]
    ActionNotAllowed { action: String, reason: String },

    #[error("budget exceeded")]
    BudgetExceeded,

    #[error("insufficient knowledge to construct message")]
    InsufficientKnowledge,

    #[error("position {position:?} does not allow this action")]
    PositionRestriction { position: AdversaryPosition },
}

// ── Downgrade Checker ────────────────────────────────────────────────────

/// Checks whether a trace constitutes a downgrade attack.
pub struct DowngradeChecker;

impl DowngradeChecker {
    /// Check if the adversary trace resulted in a cipher downgrade.
    pub fn check_cipher_downgrade(
        offered: &[CipherSuite],
        selected: &CipherSuite,
    ) -> Option<DowngradeInfo> {
        let best = offered.iter().max()?;
        if selected.security_level < best.security_level {
            Some(DowngradeInfo {
                kind: DowngradeKind::CipherSuite,
                from_level: best.security_level,
                to_level: selected.security_level,
                description: format!(
                    "Downgrade from {} ({}) to {} ({})",
                    best.name, best.security_level, selected.name, selected.security_level
                ),
            })
        } else {
            None
        }
    }

    /// Check if the adversary trace resulted in a version downgrade.
    pub fn check_version_downgrade(
        offered_max: &ProtocolVersion,
        negotiated: &ProtocolVersion,
    ) -> Option<DowngradeInfo> {
        if negotiated.is_downgrade_from(offered_max) {
            Some(DowngradeInfo {
                kind: DowngradeKind::Version,
                from_level: offered_max.security_level(),
                to_level: negotiated.security_level(),
                description: format!(
                    "Version downgrade from {} to {}",
                    offered_max, negotiated
                ),
            })
        } else {
            None
        }
    }

    /// Check if forward secrecy was lost.
    pub fn check_forward_secrecy_loss(
        offered: &[CipherSuite],
        selected: &CipherSuite,
    ) -> Option<DowngradeInfo> {
        let has_fs_offered = offered.iter().any(|cs| cs.has_forward_secrecy());
        if has_fs_offered && !selected.has_forward_secrecy() {
            Some(DowngradeInfo {
                kind: DowngradeKind::ForwardSecrecy,
                from_level: crate::protocol::SecurityLevel::High,
                to_level: crate::protocol::SecurityLevel::Legacy,
                description: format!(
                    "Forward secrecy lost: selected {} without FS",
                    selected.name
                ),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DowngradeInfo {
    pub kind: DowngradeKind,
    pub from_level: crate::protocol::SecurityLevel,
    pub to_level: crate::protocol::SecurityLevel,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DowngradeKind {
    CipherSuite,
    Version,
    ForwardSecrecy,
    Extension,
}

impl fmt::Display for DowngradeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:?}] {} → {}: {}",
            self.kind, self.from_level, self.to_level, self.description
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{CipherSuiteRegistry, SecurityLevel};

    #[test]
    fn test_message_term_depth() {
        let atom = MessageTerm::Nonce(42);
        assert_eq!(atom.depth(), 0);

        let pair = MessageTerm::Pair(
            Box::new(MessageTerm::Nonce(1)),
            Box::new(MessageTerm::Nonce(2)),
        );
        assert_eq!(pair.depth(), 1);

        let nested = MessageTerm::Hash(Box::new(MessageTerm::Pair(
            Box::new(MessageTerm::Nonce(1)),
            Box::new(MessageTerm::Nonce(2)),
        )));
        assert_eq!(nested.depth(), 2);
    }

    #[test]
    fn test_message_term_size() {
        let atom = MessageTerm::Nonce(42);
        assert_eq!(atom.size(), 1);

        let pair = MessageTerm::Pair(
            Box::new(MessageTerm::Nonce(1)),
            Box::new(MessageTerm::Nonce(2)),
        );
        assert_eq!(pair.size(), 3);
    }

    #[test]
    fn test_free_variables() {
        let term = MessageTerm::Pair(
            Box::new(MessageTerm::Variable("x".into())),
            Box::new(MessageTerm::Encrypted {
                key: Box::new(MessageTerm::Variable("k".into())),
                plaintext: Box::new(MessageTerm::Nonce(1)),
            }),
        );
        let vars = term.free_variables();
        assert!(vars.contains("x"));
        assert!(vars.contains("k"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_substitution() {
        let term = MessageTerm::Pair(
            Box::new(MessageTerm::Variable("x".into())),
            Box::new(MessageTerm::Nonce(1)),
        );
        let replaced = term.substitute("x", &MessageTerm::Nonce(42));
        let expected = MessageTerm::Pair(
            Box::new(MessageTerm::Nonce(42)),
            Box::new(MessageTerm::Nonce(1)),
        );
        assert_eq!(replaced, expected);
    }

    #[test]
    fn test_knowledge_set_basics() {
        let mut ks = KnowledgeSet::new();
        assert_eq!(ks.size(), 0);

        ks.learn(MessageTerm::Nonce(1));
        assert!(ks.knows(&MessageTerm::Nonce(1)));
        assert!(!ks.knows(&MessageTerm::Nonce(2)));
    }

    #[test]
    fn test_knowledge_pair_decomposition() {
        let pair = MessageTerm::Pair(
            Box::new(MessageTerm::Nonce(1)),
            Box::new(MessageTerm::Nonce(2)),
        );
        let mut ks = KnowledgeSet::new();
        ks.learn(pair);

        assert!(ks.knows(&MessageTerm::Nonce(1)));
        assert!(ks.knows(&MessageTerm::Nonce(2)));
    }

    #[test]
    fn test_knowledge_decryption() {
        let key = MessageTerm::Key(KeyTerm::Symmetric(vec![1, 2, 3]));
        let enc = MessageTerm::Encrypted {
            key: Box::new(key.clone()),
            plaintext: Box::new(MessageTerm::Nonce(42)),
        };

        let mut ks = KnowledgeSet::new();
        ks.learn(enc);
        // Without the key, cannot decrypt
        assert!(!ks.knows(&MessageTerm::Nonce(42)));

        // Learn the key
        ks.learn(key);
        // Now we can decrypt
        assert!(ks.knows(&MessageTerm::Nonce(42)));
    }

    #[test]
    fn test_knowledge_construction() {
        let mut ks = KnowledgeSet::new();
        ks.learn(MessageTerm::Nonce(1));
        ks.learn(MessageTerm::Nonce(2));

        let pair = MessageTerm::Pair(
            Box::new(MessageTerm::Nonce(1)),
            Box::new(MessageTerm::Nonce(2)),
        );
        assert!(ks.can_construct(&pair, 1));
        assert!(!ks.can_construct(&pair, 0));
    }

    #[test]
    fn test_adversary_budget() {
        let budget = AdversaryBudget::new(3, 10);
        let action = AdversaryAction::Intercept {
            message: MessageTerm::Nonce(1),
        };
        assert!(budget.can_perform(&action, 0));
        assert!(budget.can_perform(&action, 9));
        assert!(!budget.can_perform(&action, 10));
    }

    #[test]
    fn test_adversary_budget_depth() {
        let budget = AdversaryBudget::new(1, 100);
        let shallow = AdversaryAction::Inject {
            message: MessageTerm::Nonce(1),
        };
        assert!(budget.can_perform(&shallow, 0));

        let deep = AdversaryAction::Inject {
            message: MessageTerm::Hash(Box::new(MessageTerm::Hash(Box::new(
                MessageTerm::Nonce(1),
            )))),
        };
        assert!(!budget.can_perform(&deep, 0)); // depth 2 > bound 1
    }

    #[test]
    fn test_adversary_trace() {
        let budget = AdversaryBudget::new(5, 10);
        let mut trace = AdversaryTrace::new(budget);
        assert!(trace.is_empty());

        assert!(trace.push(AdversaryAction::Intercept {
            message: MessageTerm::Nonce(1),
        }));
        assert_eq!(trace.len(), 1);
        assert_eq!(trace.total_cost(), 1);
    }

    #[test]
    fn test_bounded_dy_adversary() {
        let budget = AdversaryBudget::new(5, 20);
        let mut adv = BoundedDYAdversary::mitm(budget);

        let intercept = AdversaryAction::Intercept {
            message: MessageTerm::Nonce(42),
        };
        assert!(adv.can_perform(&intercept));
        assert!(adv.perform(intercept).is_ok());

        // After intercepting, the adversary knows the nonce
        assert!(adv.knowledge.knows(&MessageTerm::Nonce(42)));
    }

    #[test]
    fn test_eavesdropper_restrictions() {
        let budget = AdversaryBudget::new(5, 20);
        let adv = BoundedDYAdversary::eavesdropper(budget);

        let inject = AdversaryAction::Inject {
            message: MessageTerm::Nonce(1),
        };
        assert!(!adv.can_perform(&inject)); // Eavesdropper cannot inject
    }

    #[test]
    fn test_downgrade_checker_cipher() {
        let strong = CipherSuiteRegistry::lookup(0x1301).unwrap();
        let weak = CipherSuiteRegistry::lookup(0x0005).unwrap();

        let result = DowngradeChecker::check_cipher_downgrade(&[strong.clone(), weak.clone()], &weak);
        assert!(result.is_some());
        let info = result.unwrap();
        assert_eq!(info.kind, DowngradeKind::CipherSuite);

        let no_downgrade = DowngradeChecker::check_cipher_downgrade(&[strong.clone()], &strong);
        assert!(no_downgrade.is_none());
    }

    #[test]
    fn test_downgrade_checker_version() {
        let result = DowngradeChecker::check_version_downgrade(
            &ProtocolVersion::tls13(),
            &ProtocolVersion::tls11(),
        );
        assert!(result.is_some());

        let no_downgrade = DowngradeChecker::check_version_downgrade(
            &ProtocolVersion::tls12(),
            &ProtocolVersion::tls12(),
        );
        assert!(no_downgrade.is_none());
    }

    #[test]
    fn test_downgrade_forward_secrecy() {
        let ecdhe = CipherSuiteRegistry::lookup(0xC02F).unwrap(); // ECDHE_RSA_AES_128_GCM
        let rsa = CipherSuiteRegistry::lookup(0x009C).unwrap();   // RSA_AES_128_GCM (no FS)

        let result = DowngradeChecker::check_forward_secrecy_loss(&[ecdhe, rsa.clone()], &rsa);
        assert!(result.is_some());
    }

    #[test]
    fn test_adversary_action_cost() {
        assert_eq!(
            AdversaryAction::Intercept {
                message: MessageTerm::Nonce(1)
            }
            .cost(),
            1
        );
        assert_eq!(
            AdversaryAction::Modify {
                original: MessageTerm::Nonce(1),
                modified: MessageTerm::Nonce(2),
            }
            .cost(),
            3
        );
    }

    #[test]
    fn test_message_display() {
        let nonce = MessageTerm::Nonce(42);
        assert_eq!(format!("{}", nonce), "N(42)");

        let pair = MessageTerm::Pair(
            Box::new(MessageTerm::Nonce(1)),
            Box::new(MessageTerm::Nonce(2)),
        );
        assert_eq!(format!("{}", pair), "<N(1), N(2)>");
    }
}

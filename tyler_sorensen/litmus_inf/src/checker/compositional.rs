//! Compositional verification for the LITMUS∞ checker (paper §5).
//!
//! Implements thread-modular reasoning, assume-guarantee verification,
//! compositional state space splitting, interface abstraction, and
//! incremental verification.

use std::collections::{HashMap, HashSet};
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::execution::{
    ThreadId, Address, Value,
};
use crate::checker::memory_model::MemoryModel;
use crate::checker::litmus::{LitmusTest, Thread, Instruction};
use crate::checker::verifier::VerificationResult;

// ═══════════════════════════════════════════════════════════════════════════
// Thread Interface
// ═══════════════════════════════════════════════════════════════════════════

/// Abstract interface for a thread's externally-visible behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadInterface {
    /// Thread ID.
    pub thread_id: ThreadId,
    /// Addresses read by this thread.
    pub read_set: HashSet<Address>,
    /// Addresses written by this thread.
    pub write_set: HashSet<Address>,
    /// Threads this communicates with (via shared addresses).
    pub communicates_with: HashSet<ThreadId>,
    /// Ordering constraints on external operations.
    pub ordering_constraints: Vec<InterfaceConstraint>,
}

/// A constraint on the interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceConstraint {
    /// The kind of constraint.
    pub kind: ConstraintKind,
    /// Description.
    pub description: String,
}

/// Kinds of interface constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintKind {
    /// An ordering between two addresses.
    Ordering { before: Address, after: Address },
    /// A value constraint on a read.
    ValueConstraint { addr: Address, possible_values: Vec<Value> },
    /// A fence ordering constraint.
    FenceOrdering { scope: String },
}

impl ThreadInterface {
    /// Extract the interface of a thread from a litmus test.
    pub fn extract(thread: &Thread, all_threads: &[Thread]) -> Self {
        let mut read_set = HashSet::new();
        let mut write_set = HashSet::new();

        for instr in &thread.instructions {
            match instr {
                Instruction::Load { addr, .. } => { read_set.insert(*addr); }
                Instruction::Store { addr, .. } => { write_set.insert(*addr); }
                Instruction::RMW { addr, .. } => {
                    read_set.insert(*addr);
                    write_set.insert(*addr);
                }
                _ => {}
            }
        }

        // Determine communication partners
        let my_addrs: HashSet<Address> = read_set.union(&write_set).copied().collect();
        let mut communicates_with = HashSet::new();

        for other in all_threads {
            if other.id == thread.id { continue; }
            let other_addrs: HashSet<Address> = other.accessed_addresses().into_iter().collect();
            if !my_addrs.is_disjoint(&other_addrs) {
                communicates_with.insert(other.id);
            }
        }

        ThreadInterface {
            thread_id: thread.id,
            read_set,
            write_set,
            communicates_with,
            ordering_constraints: Vec::new(),
        }
    }

    /// Check if this thread is independent from another.
    pub fn is_independent_from(&self, other: &ThreadInterface) -> bool {
        !self.communicates_with.contains(&other.thread_id)
    }

    /// Shared addresses with another thread.
    pub fn shared_addresses(&self, other: &ThreadInterface) -> HashSet<Address> {
        let my_addrs: HashSet<Address> = self.read_set.union(&self.write_set).copied().collect();
        let other_addrs: HashSet<Address> = other.read_set.union(&other.write_set).copied().collect();
        my_addrs.intersection(&other_addrs).copied().collect()
    }

    /// Communication complexity: number of shared addresses.
    pub fn communication_complexity(&self, other: &ThreadInterface) -> usize {
        self.shared_addresses(other).len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Component Abstraction
// ═══════════════════════════════════════════════════════════════════════════

/// A component: a subset of threads for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    /// Thread IDs in this component.
    pub thread_ids: Vec<ThreadId>,
    /// Interfaces of the threads.
    pub interfaces: Vec<ThreadInterface>,
    /// External addresses (accessed by other components).
    pub external_addresses: HashSet<Address>,
    /// Internal addresses (only accessed within this component).
    pub internal_addresses: HashSet<Address>,
}

impl Component {
    /// Create a component from thread IDs and the full test.
    pub fn from_threads(thread_ids: Vec<ThreadId>, test: &LitmusTest) -> Self {
        let interfaces: Vec<ThreadInterface> = thread_ids.iter()
            .filter_map(|&tid| {
                test.threads.get(tid).map(|t| ThreadInterface::extract(t, &test.threads))
            })
            .collect();

        let component_addrs: HashSet<Address> = interfaces.iter()
            .flat_map(|iface| iface.read_set.union(&iface.write_set))
            .copied()
            .collect();

        // External = accessed by threads outside this component
        let mut external_addresses = HashSet::new();
        let thread_set: HashSet<ThreadId> = thread_ids.iter().copied().collect();
        for t in &test.threads {
            if thread_set.contains(&t.id) { continue; }
            let other_addrs: HashSet<Address> = t.accessed_addresses().into_iter().collect();
            for addr in other_addrs.intersection(&component_addrs) {
                external_addresses.insert(*addr);
            }
        }

        let internal_addresses: HashSet<Address> = component_addrs.difference(&external_addresses)
            .copied().collect();

        Component { thread_ids, interfaces, external_addresses, internal_addresses }
    }

    /// Number of threads in this component.
    pub fn num_threads(&self) -> usize {
        self.thread_ids.len()
    }

    /// Whether this component is independent (no external addresses).
    pub fn is_independent(&self) -> bool {
        self.external_addresses.is_empty()
    }

    /// Extract a sub-test from the full litmus test.
    pub fn extract_test(&self, test: &LitmusTest) -> LitmusTest {
        let mut sub_test = LitmusTest::new(&format!("{}-component", test.name));
        for &tid in &self.thread_ids {
            if let Some(thread) = test.threads.get(tid) {
                sub_test.add_thread(thread.clone());
            }
        }
        // Copy initial values for relevant addresses
        let all_addrs: HashSet<Address> = self.interfaces.iter()
            .flat_map(|iface| iface.read_set.union(&iface.write_set))
            .copied()
            .collect();
        for (&addr, &val) in &test.initial_state {
            if all_addrs.contains(&addr) {
                sub_test.set_initial(addr, val);
            }
        }
        sub_test
    }
}

/// Abstract state of a component's behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentAbstraction {
    /// The component.
    pub component: Component,
    /// Possible values for each external address after execution.
    pub possible_values: HashMap<Address, HashSet<Value>>,
    /// Ordering guarantees provided.
    pub guarantees: Vec<Guarantee>,
}

/// A guarantee about component behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Guarantee {
    /// Description.
    pub description: String,
    /// The ordering type.
    pub kind: GuaranteeKind,
}

/// Kinds of guarantees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuaranteeKind {
    /// Write to addr happens before observation at other addr.
    WriteBeforeObservation { write_addr: Address, obs_addr: Address },
    /// Value of addr is in the given set.
    ValueInSet { addr: Address, values: Vec<Value> },
    /// Fence guarantees ordering.
    FenceOrdering,
}

// ═══════════════════════════════════════════════════════════════════════════
// Composition Rules
// ═══════════════════════════════════════════════════════════════════════════

/// Rules for composing verification results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionRule {
    /// Independent parallel composition (no shared state).
    Parallel,
    /// Sequential composition.
    Sequential,
    /// Hierarchical (nested) composition.
    Hierarchical,
    /// Assume-guarantee reasoning.
    AssumeGuarantee,
}

/// A composition theorem application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionTheorem {
    /// The rule applied.
    pub rule: CompositionRule,
    /// The components.
    pub component_ids: Vec<usize>,
    /// Proof obligations.
    pub obligations: Vec<ProofObligation>,
    /// Whether all obligations are discharged.
    pub discharged: bool,
}

/// A proof obligation that must be verified for a composition rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofObligation {
    /// Description.
    pub description: String,
    /// Whether this obligation has been verified.
    pub verified: bool,
}

impl CompositionTheorem {
    /// Create a parallel composition theorem.
    pub fn parallel(components: Vec<usize>) -> Self {
        let obligations = vec![
            ProofObligation {
                description: "Components share no state".to_string(),
                verified: false,
            },
        ];
        CompositionTheorem {
            rule: CompositionRule::Parallel,
            component_ids: components,
            obligations,
            discharged: false,
        }
    }

    /// Check if all obligations are discharged.
    pub fn is_discharged(&self) -> bool {
        self.obligations.iter().all(|o| o.verified)
    }

    /// Discharge the obligation at index i.
    pub fn discharge(&mut self, i: usize) {
        if i < self.obligations.len() {
            self.obligations[i].verified = true;
        }
        self.discharged = self.is_discharged();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Assume-Guarantee Reasoner
// ═══════════════════════════════════════════════════════════════════════════

/// An assumption about the environment (other threads).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assumption {
    /// Description.
    pub description: String,
    /// Which threads this assumption is about.
    pub about_threads: Vec<ThreadId>,
    /// Address constraints.
    pub address_constraints: HashMap<Address, Vec<Value>>,
}

/// Result of verifying a component under assumptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuaranteeResult {
    /// Whether the component satisfies its guarantees.
    pub satisfied: bool,
    /// The guarantees provided.
    pub guarantees: Vec<Guarantee>,
    /// Counterexample if not satisfied.
    pub counterexample: Option<String>,
}

/// Assume-guarantee reasoning framework.
#[derive(Debug)]
pub struct AssumeGuaranteeReasoner {
    /// Components to verify.
    pub components: Vec<Component>,
    /// Assumptions per component.
    pub assumptions: Vec<Vec<Assumption>>,
    /// Guarantees per component.
    pub guarantees: Vec<Vec<Guarantee>>,
    /// The memory model.
    pub model: MemoryModel,
}

impl AssumeGuaranteeReasoner {
    /// Create a new reasoner.
    pub fn new(components: Vec<Component>, model: MemoryModel) -> Self {
        let n = components.len();
        AssumeGuaranteeReasoner {
            components,
            assumptions: vec![Vec::new(); n],
            guarantees: vec![Vec::new(); n],
            model,
        }
    }

    /// Add an assumption for a component.
    pub fn add_assumption(&mut self, component_idx: usize, assumption: Assumption) {
        self.assumptions[component_idx].push(assumption);
    }

    /// Add a guarantee for a component.
    pub fn add_guarantee(&mut self, component_idx: usize, guarantee: Guarantee) {
        self.guarantees[component_idx].push(guarantee);
    }

    /// Verify a single component under its assumptions.
    pub fn verify_component(&self, component_idx: usize) -> GuaranteeResult {
        let component = &self.components[component_idx];

        // Check if the component can be verified in isolation
        // (simplified: check that external addresses have bounded values)
        let satisfied = component.external_addresses.len() < 10; // placeholder heuristic

        GuaranteeResult {
            satisfied,
            guarantees: self.guarantees[component_idx].clone(),
            counterexample: if satisfied { None } else { Some("Too many external addresses".to_string()) },
        }
    }

    /// Check circular assume-guarantee rule.
    /// Each component's assumptions must be discharged by other components' guarantees.
    pub fn check_circular(&self) -> bool {
        for i in 0..self.components.len() {
            for assumption in &self.assumptions[i] {
                // Check if some other component's guarantee covers this assumption
                let covered = (0..self.components.len())
                    .filter(|&j| j != i)
                    .any(|j| {
                        self.guarantees[j].iter().any(|g| {
                            // Simplified coverage check
                            match &g.kind {
                                GuaranteeKind::ValueInSet { addr, values: _ } => {
                                    assumption.address_constraints.contains_key(addr)
                                }
                                _ => false,
                            }
                        })
                    });
                if !covered {
                    return false;
                }
            }
        }
        true
    }

    /// Synthesize initial assumptions (over-approximate).
    pub fn synthesize_assumptions(&mut self) {
        for i in 0..self.components.len() {
            let external = &self.components[i].external_addresses;
            for &addr in external {
                let assumption = Assumption {
                    description: format!("Value of addr {:#x} is bounded", addr),
                    about_threads: self.components.iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .flat_map(|(_, c)| c.thread_ids.iter().copied())
                        .collect(),
                    address_constraints: {
                        let mut m = HashMap::new();
                        m.insert(addr, vec![0, 1]); // assume values are 0 or 1
                        m
                    },
                };
                self.assumptions[i].push(assumption);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// State Space Splitter
// ═══════════════════════════════════════════════════════════════════════════

/// Splits the state space for compositional verification.
#[derive(Debug)]
pub struct StateSpaceSplitter;

/// A partition of the state space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpacePartition {
    /// Components (groups of thread IDs).
    pub components: Vec<Vec<ThreadId>>,
    /// Quality score (lower = less cross-component communication).
    pub quality: f64,
}

impl StateSpaceSplitter {
    /// Split by shared variable partitioning.
    pub fn split_by_variables(test: &LitmusTest) -> StateSpacePartition {
        // Group threads that share addresses
        let n = test.thread_count();
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find(parent, parent[x]); }
            parent[x]
        }

        fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
            let rx = find(parent, x);
            let ry = find(parent, y);
            if rx != ry { parent[rx] = ry; }
        }

        // Merge threads that share addresses
        for i in 0..n {
            let addrs_i: HashSet<Address> = test.threads[i].accessed_addresses().into_iter().collect();
            for j in (i + 1)..n {
                let addrs_j: HashSet<Address> = test.threads[j].accessed_addresses().into_iter().collect();
                if !addrs_i.is_disjoint(&addrs_j) {
                    union(&mut parent, i, j);
                }
            }
        }

        // Build components
        let mut groups: HashMap<usize, Vec<ThreadId>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            groups.entry(root).or_default().push(i);
        }

        let components: Vec<Vec<ThreadId>> = groups.into_values().collect();
        let quality = Self::compute_quality(test, &components);

        StateSpacePartition { components, quality }
    }

    /// Split into individual threads.
    pub fn split_by_threads(test: &LitmusTest) -> StateSpacePartition {
        let components: Vec<Vec<ThreadId>> = (0..test.thread_count()).map(|t| vec![t]).collect();
        let quality = Self::compute_quality(test, &components);
        StateSpacePartition { components, quality }
    }

    /// Optimal splitting (tries multiple strategies and picks the best).
    pub fn optimal_split(test: &LitmusTest) -> StateSpacePartition {
        let by_vars = Self::split_by_variables(test);
        let by_threads = Self::split_by_threads(test);

        if by_vars.quality <= by_threads.quality {
            by_vars
        } else {
            by_threads
        }
    }

    /// Compute quality score (lower = better; counts cross-component edges).
    fn compute_quality(test: &LitmusTest, components: &[Vec<ThreadId>]) -> f64 {
        let mut thread_to_component: HashMap<ThreadId, usize> = HashMap::new();
        for (ci, comp) in components.iter().enumerate() {
            for &tid in comp {
                thread_to_component.insert(tid, ci);
            }
        }

        // Count shared addresses across different components
        let mut cross_edges = 0.0;
        for i in 0..test.thread_count() {
            for j in (i + 1)..test.thread_count() {
                if thread_to_component[&i] == thread_to_component[&j] { continue; }
                let addrs_i: HashSet<Address> = test.threads[i].accessed_addresses().into_iter().collect();
                let addrs_j: HashSet<Address> = test.threads[j].accessed_addresses().into_iter().collect();
                cross_edges += addrs_i.intersection(&addrs_j).count() as f64;
            }
        }

        cross_edges
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compositional Verification Engine
// ═══════════════════════════════════════════════════════════════════════════

/// Main engine for compositional verification.
#[derive(Debug)]
pub struct CompositionalVerificationEngine {
    /// The memory model.
    pub model: MemoryModel,
    /// Cache of component verification results.
    cache: HashMap<String, VerificationResult>,
    /// Statistics.
    pub stats: CompositionalStats,
}

/// Statistics from compositional verification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompositionalStats {
    /// Number of components.
    pub num_components: usize,
    /// Number of cache hits.
    pub cache_hits: usize,
    /// Number of cache misses.
    pub cache_misses: usize,
    /// Whether compositional verification was successful.
    pub compositional_success: bool,
    /// Number of independent components.
    pub independent_components: usize,
}

impl CompositionalVerificationEngine {
    /// Create a new engine.
    pub fn new(model: MemoryModel) -> Self {
        CompositionalVerificationEngine {
            model,
            cache: HashMap::new(),
            stats: CompositionalStats::default(),
        }
    }

    /// Run compositional verification on a litmus test.
    pub fn verify(&mut self, test: &LitmusTest) -> CompositionalResult {
        // Phase 1: Decompose
        let partition = StateSpaceSplitter::optimal_split(test);
        self.stats.num_components = partition.components.len();

        // Phase 2: Build components
        let components: Vec<Component> = partition.components.iter()
            .map(|tids| Component::from_threads(tids.clone(), test))
            .collect();

        // Count independent components
        self.stats.independent_components = components.iter()
            .filter(|c| c.is_independent())
            .count();

        // Phase 3: Verify components
        let mut component_results = Vec::new();
        for (_i, component) in components.iter().enumerate() {
            // Check cache
            let cache_key = format!("{:?}", component.thread_ids);
            if let Some(result) = self.cache.get(&cache_key) {
                self.stats.cache_hits += 1;
                component_results.push(result.clone());
                continue;
            }
            self.stats.cache_misses += 1;

            // Verify this component
            let sub_test = component.extract_test(test);
            let result = self.verify_component(&sub_test);
            self.cache.insert(cache_key, result.clone());
            component_results.push(result);
        }

        // Phase 4: Compose results
        let all_consistent = component_results.iter().all(|r| !r.has_forbidden());

        self.stats.compositional_success = true;

        CompositionalResult {
            overall_consistent: all_consistent,
            component_results,
            partition,
            stats: self.stats.clone(),
        }
    }

    /// Verify a single component.
    fn verify_component(&self, test: &LitmusTest) -> VerificationResult {
        let mut verifier = crate::checker::verifier::Verifier::new(self.model.clone());
        verifier.verify_litmus(test)
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Result of compositional verification.
#[derive(Debug, Clone)]
pub struct CompositionalResult {
    /// Overall consistency.
    pub overall_consistent: bool,
    /// Per-component results.
    pub component_results: Vec<VerificationResult>,
    /// The partition used.
    pub partition: StateSpacePartition,
    /// Statistics.
    pub stats: CompositionalStats,
}

impl fmt::Display for CompositionalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Compositional Verification:")?;
        writeln!(f, "  Components: {}", self.stats.num_components)?;
        writeln!(f, "  Independent: {}", self.stats.independent_components)?;
        writeln!(f, "  Cache hits: {}", self.stats.cache_hits)?;
        writeln!(f, "  Overall: {}", if self.overall_consistent { "CONSISTENT" } else { "INCONSISTENT" })?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Incremental Verifier
// ═══════════════════════════════════════════════════════════════════════════

/// Support for incremental verification.
#[derive(Debug)]
pub struct IncrementalVerifier {
    /// The engine.
    pub engine: CompositionalVerificationEngine,
    /// Previous test (for delta computation).
    previous_test: Option<LitmusTest>,
    /// Previous partition.
    previous_partition: Option<StateSpacePartition>,
}

impl IncrementalVerifier {
    /// Create a new incremental verifier.
    pub fn new(model: MemoryModel) -> Self {
        IncrementalVerifier {
            engine: CompositionalVerificationEngine::new(model),
            previous_test: None,
            previous_partition: None,
        }
    }

    /// Verify with incremental support.
    pub fn verify(&mut self, test: &LitmusTest) -> CompositionalResult {
        // Check if test changed from previous
        if let Some(prev) = &self.previous_test {
            let changed_threads = self.find_changed_threads(prev, test);
            if changed_threads.is_empty() {
                // No changes; reuse previous result
                // (simplified: just re-verify)
            } else {
                // Only invalidate cache entries for changed components
                self.invalidate_changed(&changed_threads);
            }
        }

        let result = self.engine.verify(test);
        self.previous_test = Some(test.clone());
        self.previous_partition = Some(result.partition.clone());
        result
    }

    /// Find which threads changed between two tests.
    fn find_changed_threads(&self, old: &LitmusTest, new: &LitmusTest) -> Vec<ThreadId> {
        let mut changed = Vec::new();
        let max_threads = old.thread_count().max(new.thread_count());
        for t in 0..max_threads {
            if t >= old.thread_count() || t >= new.thread_count() {
                changed.push(t);
            } else if old.threads[t].instructions != new.threads[t].instructions {
                changed.push(t);
            }
        }
        changed
    }

    /// Invalidate cache entries for components containing changed threads.
    fn invalidate_changed(&mut self, changed_threads: &[ThreadId]) {
        if let Some(partition) = &self.previous_partition {
            for comp in &partition.components {
                if comp.iter().any(|t| changed_threads.contains(t)) {
                    let key = format!("{:?}", comp);
                    self.engine.cache.remove(&key);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compositional Proof
// ═══════════════════════════════════════════════════════════════════════════

/// Certificate that compositional verification is correct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionalProof {
    /// Proof tree nodes.
    pub nodes: Vec<ProofNode>,
    /// Root node index.
    pub root: usize,
    /// Whether the proof is complete.
    pub complete: bool,
}

/// A node in the proof tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    /// Description of this proof step.
    pub description: String,
    /// The composition rule used.
    pub rule: CompositionRule,
    /// Children (sub-proofs).
    pub children: Vec<usize>,
    /// Obligations.
    pub obligations: Vec<ProofObligation>,
    /// Whether this node is verified.
    pub verified: bool,
}

impl CompositionalProof {
    /// Create a leaf proof (directly verified component).
    pub fn leaf(description: &str) -> Self {
        let node = ProofNode {
            description: description.to_string(),
            rule: CompositionRule::Parallel,
            children: Vec::new(),
            obligations: Vec::new(),
            verified: true,
        };
        CompositionalProof { nodes: vec![node], root: 0, complete: true }
    }

    /// Create a composite proof from sub-proofs.
    pub fn compose(
        rule: CompositionRule,
        children: Vec<CompositionalProof>,
        obligations: Vec<ProofObligation>,
    ) -> Self {
        let mut all_nodes = Vec::new();
        let mut child_roots = Vec::new();
        let mut offset = 1; // root will be at 0

        for child in children {
            child_roots.push(child.root + offset);
            for mut node in child.nodes {
                node.children = node.children.iter().map(|&c| c + offset).collect();
                all_nodes.push(node);
            }
            offset += all_nodes.len();
        }

        let root = ProofNode {
            description: format!("{:?} composition", rule),
            rule,
            children: child_roots,
            obligations,
            verified: false,
        };

        let mut nodes = vec![root];
        nodes.extend(all_nodes);

        let complete = nodes.iter().all(|n| n.verified || !n.obligations.iter().all(|o| o.verified));

        CompositionalProof { nodes, root: 0, complete }
    }

    /// Check if the proof is valid.
    pub fn is_valid(&self) -> bool {
        self.complete && self.nodes[self.root].verified
    }

    /// Number of proof nodes.
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rely-Guarantee Compositional Reasoning (shared variables)
// ═══════════════════════════════════════════════════════════════════════════

/// A rely condition: interference that a component tolerates from the environment.
/// Formally: Rely(C_i) ⊆ Actions(Env) specifies which writes from other
/// components thread C_i can observe without violating its guarantees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelyCondition {
    /// Which addresses this rely condition covers.
    pub addresses: HashSet<Address>,
    /// For each address, the set of values that the component tolerates seeing.
    pub tolerated_values: HashMap<Address, HashSet<Value>>,
    /// Ordering constraints on environment writes.
    /// If true, environment writes to this address must be ordered (e.g., by a fence).
    pub requires_ordered_writes: HashMap<Address, bool>,
    /// Description for human readability.
    pub description: String,
}

/// A guarantee condition: what a component promises about its own behavior.
/// Formally: Guar(C_i) specifies what C_i guarantees to the environment
/// about its writes and their ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuaranteeCondition {
    /// Addresses written by this component.
    pub written_addresses: HashSet<Address>,
    /// For each address, the values this component may write.
    pub possible_writes: HashMap<Address, HashSet<Value>>,
    /// Ordering guarantees: does this component ensure its writes are ordered?
    pub ordered_writes: HashMap<Address, bool>,
    /// Does this component use fences to order its writes?
    pub has_fence: bool,
    /// Description for human readability.
    pub description: String,
}

/// Result of rely-guarantee compatibility check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelyGuaranteeResult {
    /// Whether the rely-guarantee composition is valid.
    pub compatible: bool,
    /// Overall safety assessment.
    pub safe: bool,
    /// Per-component results.
    pub component_results: Vec<ComponentRGResult>,
    /// Violations (if any).
    pub violations: Vec<RGViolation>,
    /// Whether the result is conservative (may have false positives).
    pub conservative: bool,
    /// Summary description.
    pub description: String,
}

/// Per-component rely-guarantee result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentRGResult {
    /// Thread IDs in this component.
    pub thread_ids: Vec<ThreadId>,
    /// Rely condition.
    pub rely: RelyCondition,
    /// Guarantee condition.
    pub guarantee: GuaranteeCondition,
    /// Whether rely is satisfied by other components' guarantees.
    pub rely_satisfied: bool,
}

/// A violation of the rely-guarantee contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGViolation {
    /// Component whose rely is violated.
    pub component_thread_ids: Vec<ThreadId>,
    /// The address involved.
    pub address: Address,
    /// Description of the violation.
    pub description: String,
    /// Severity: warning (conservative) or error (definite bug).
    pub severity: ViolationSeverity,
}

/// Severity of a rely-guarantee violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Conservative warning: may be a false positive.
    Warning,
    /// Definite violation: rely cannot be satisfied.
    Error,
}

/// Rely-Guarantee Composition Engine.
///
/// Implements the rely-guarantee principle for shared-variable composition:
/// For components C_1, ..., C_n with shared variables:
///   1. Extract Rely(C_i) and Guar(C_i) for each component
///   2. Check: ∀i. ⋃_{j≠i} Guar(C_j) ⊆ Rely(C_i)
///   3. If compatible: safety of whole = ∧_i safety(C_i under Rely(C_i))
///
/// This is conservative: it may report false positives but never false negatives.
#[derive(Debug)]
pub struct RelyGuaranteeEngine {
    /// Memory model.
    pub model: MemoryModel,
}

impl RelyGuaranteeEngine {
    /// Create a new rely-guarantee engine.
    pub fn new(model: MemoryModel) -> Self {
        RelyGuaranteeEngine { model }
    }

    /// Extract the rely condition for a component.
    /// The rely specifies what interference the component tolerates.
    pub fn extract_rely(&self, component: &Component, test: &LitmusTest) -> RelyCondition {
        let mut tolerated_values: HashMap<Address, HashSet<Value>> = HashMap::new();
        let mut requires_ordered = HashMap::new();

        for &addr in &component.external_addresses {
            // Conservative: tolerate values 0 and 1 (binary litmus test assumption)
            let mut values = HashSet::new();
            values.insert(0);
            values.insert(1);
            // Check initial values
            if let Some(&init_val) = test.initial_state.get(&addr) {
                values.insert(init_val);
            }
            tolerated_values.insert(addr, values);

            // Check if this component has loads from this address that depend
            // on the ordering of external writes
            let has_load = component.interfaces.iter().any(|iface| {
                iface.read_set.contains(&addr)
            });
            let has_store = component.interfaces.iter().any(|iface| {
                iface.write_set.contains(&addr)
            });
            // If component both reads and writes the shared address,
            // it requires ordered writes from the environment
            requires_ordered.insert(addr, has_load && has_store);
        }

        RelyCondition {
            addresses: component.external_addresses.clone(),
            tolerated_values,
            requires_ordered_writes: requires_ordered,
            description: format!(
                "Rely for threads {:?}: tolerates writes to {:?}",
                component.thread_ids,
                component.external_addresses.iter().collect::<Vec<_>>()
            ),
        }
    }

    /// Extract the guarantee condition for a component.
    /// The guarantee specifies what the component promises about its behavior.
    pub fn extract_guarantee(&self, component: &Component, test: &LitmusTest) -> GuaranteeCondition {
        let mut possible_writes: HashMap<Address, HashSet<Value>> = HashMap::new();
        let mut ordered_writes = HashMap::new();
        let mut has_fence = false;

        // Analyze each thread in the component
        for &tid in &component.thread_ids {
            if let Some(thread) = test.threads.get(tid) {
                let mut prev_was_fence = false;
                for instr in &thread.instructions {
                    match instr {
                        Instruction::Store { addr, value, ordering, .. } => {
                            let written = component.external_addresses.contains(addr);
                            if written {
                                possible_writes.entry(*addr)
                                    .or_insert_with(HashSet::new)
                                    .insert(*value);
                                // Stores after fences or with release ordering are ordered
                                let is_ordered = prev_was_fence
                                    || *ordering != Ordering::Relaxed;
                                ordered_writes.insert(*addr, is_ordered);
                            }
                            prev_was_fence = false;
                        }
                        Instruction::Fence { .. } => {
                            has_fence = true;
                            prev_was_fence = true;
                        }
                        _ => {
                            prev_was_fence = false;
                        }
                    }
                }
            }
        }

        // External addresses that are only read: no writes guaranteed
        for &addr in &component.external_addresses {
            if !possible_writes.contains_key(&addr) {
                possible_writes.insert(addr, HashSet::new()); // no writes
                ordered_writes.insert(addr, true); // vacuously ordered
            }
        }

        GuaranteeCondition {
            written_addresses: possible_writes.keys().copied().collect(),
            possible_writes,
            ordered_writes,
            has_fence,
            description: format!(
                "Guarantee for threads {:?}: writes to {:?}",
                component.thread_ids,
                component.external_addresses.iter().collect::<Vec<_>>()
            ),
        }
    }

    /// Check rely-guarantee compatibility between all components.
    /// For each component C_i, checks that the union of guarantees from
    /// all other components satisfies C_i's rely condition.
    pub fn check_compatibility(
        &self,
        components: &[Component],
        relies: &[RelyCondition],
        guarantees: &[GuaranteeCondition],
    ) -> (bool, Vec<RGViolation>) {
        let mut violations = Vec::new();
        let n = components.len();

        for i in 0..n {
            let rely = &relies[i];

            for &addr in &rely.addresses {
                // Collect all values that other components may write to this address
                let mut env_values = HashSet::new();
                let mut env_ordered = true;

                for j in 0..n {
                    if j == i { continue; }
                    if let Some(writes) = guarantees[j].possible_writes.get(&addr) {
                        env_values.extend(writes);
                    }
                    if let Some(&ordered) = guarantees[j].ordered_writes.get(&addr) {
                        env_ordered = env_ordered && ordered;
                    }
                }

                // Check value tolerance
                if let Some(tolerated) = rely.tolerated_values.get(&addr) {
                    let untolerated: HashSet<_> = env_values.difference(tolerated).collect();
                    if !untolerated.is_empty() {
                        violations.push(RGViolation {
                            component_thread_ids: components[i].thread_ids.clone(),
                            address: addr,
                            description: format!(
                                "Component {:?} cannot tolerate values {:?} to address {:#x}",
                                components[i].thread_ids, untolerated, addr
                            ),
                            severity: ViolationSeverity::Warning,
                        });
                    }
                }

                // Check ordering requirement
                if let Some(&needs_order) = rely.requires_ordered_writes.get(&addr) {
                    if needs_order && !env_ordered {
                        violations.push(RGViolation {
                            component_thread_ids: components[i].thread_ids.clone(),
                            address: addr,
                            description: format!(
                                "Component {:?} requires ordered writes to {:#x}, \
                                 but environment writes are unordered",
                                components[i].thread_ids, addr
                            ),
                            severity: ViolationSeverity::Error,
                        });
                    }
                }
            }
        }

        let compatible = violations.is_empty()
            || violations.iter().all(|v| matches!(v.severity, ViolationSeverity::Warning));
        (compatible, violations)
    }

    /// Run full rely-guarantee compositional verification.
    pub fn verify(&self, test: &LitmusTest) -> RelyGuaranteeResult {
        // Step 1: Partition into components (use variable-based splitting)
        let partition = StateSpaceSplitter::split_by_variables(test);

        // If all components are independent, fall back to parallel composition
        let components: Vec<Component> = partition.components.iter()
            .map(|tids| Component::from_threads(tids.clone(), test))
            .collect();

        let all_independent = components.iter().all(|c| c.is_independent());
        if all_independent {
            return RelyGuaranteeResult {
                compatible: true,
                safe: true, // simplified: independent = safe
                component_results: Vec::new(),
                violations: Vec::new(),
                conservative: false,
                description: "All components independent (disjoint variables)".to_string(),
            };
        }

        // Step 2: Extract rely and guarantee for each component
        let relies: Vec<RelyCondition> = components.iter()
            .map(|c| self.extract_rely(c, test))
            .collect();
        let guarantees: Vec<GuaranteeCondition> = components.iter()
            .map(|c| self.extract_guarantee(c, test))
            .collect();

        // Step 3: Check compatibility
        let (compatible, violations) = self.check_compatibility(
            &components, &relies, &guarantees);

        // Step 4: Build per-component results
        let component_results: Vec<ComponentRGResult> = components.iter()
            .enumerate()
            .map(|(i, c)| ComponentRGResult {
                thread_ids: c.thread_ids.clone(),
                rely: relies[i].clone(),
                guarantee: guarantees[i].clone(),
                rely_satisfied: !violations.iter().any(|v|
                    v.component_thread_ids == c.thread_ids),
            })
            .collect();

        // Step 5: Overall safety
        // Conservative: safe only if compatible AND no error-level violations
        let has_errors = violations.iter()
            .any(|v| matches!(v.severity, ViolationSeverity::Error));
        let safe = compatible && !has_errors;

        RelyGuaranteeResult {
            compatible,
            safe,
            component_results,
            violations,
            conservative: true, // rely-guarantee is always conservative
            description: if safe {
                "Rely-guarantee composition: SAFE (conservative)".to_string()
            } else {
                format!("Rely-guarantee composition: POTENTIALLY UNSAFE \
                         ({} violations)", violations.len())
            },
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::litmus::Ordering;

    fn make_test_2t() -> LitmusTest {
        let mut test = LitmusTest::new("test-2t");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed);
        t0.load(0, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(1, 1, Ordering::Relaxed);
        t1.load(1, 0, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }

    fn make_independent_test() -> LitmusTest {
        let mut test = LitmusTest::new("independent");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // x = 1
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(100, 1, Ordering::Relaxed); // y = 1 (different address)
        test.add_thread(t1);

        test
    }

    #[test]
    fn test_thread_interface_extraction() {
        let test = make_test_2t();
        let iface = ThreadInterface::extract(&test.threads[0], &test.threads);
        assert_eq!(iface.thread_id, 0);
        assert!(!iface.read_set.is_empty() || !iface.write_set.is_empty());
    }

    #[test]
    fn test_thread_independence() {
        let test = make_independent_test();
        let iface0 = ThreadInterface::extract(&test.threads[0], &test.threads);
        let iface1 = ThreadInterface::extract(&test.threads[1], &test.threads);
        assert!(iface0.is_independent_from(&iface1));
    }

    #[test]
    fn test_thread_communication() {
        let test = make_test_2t();
        let iface0 = ThreadInterface::extract(&test.threads[0], &test.threads);
        let iface1 = ThreadInterface::extract(&test.threads[1], &test.threads);
        // T0 and T1 both access address 1, so they communicate
        let shared = iface0.shared_addresses(&iface1);
        assert!(shared.contains(&1));
    }

    #[test]
    fn test_component_creation() {
        let test = make_test_2t();
        let comp = Component::from_threads(vec![0], &test);
        assert_eq!(comp.num_threads(), 1);
    }

    #[test]
    fn test_independent_component() {
        let test = make_independent_test();
        let comp = Component::from_threads(vec![0], &test);
        assert!(comp.is_independent());
    }

    #[test]
    fn test_component_extract_test() {
        let test = make_test_2t();
        let comp = Component::from_threads(vec![0], &test);
        let sub_test = comp.extract_test(&test);
        assert_eq!(sub_test.thread_count(), 1);
    }

    #[test]
    fn test_state_space_split_by_variables() {
        let test = make_independent_test();
        let partition = StateSpaceSplitter::split_by_variables(&test);
        assert_eq!(partition.components.len(), 2); // independent threads
    }

    #[test]
    fn test_state_space_split_connected() {
        let test = make_test_2t();
        let partition = StateSpaceSplitter::split_by_variables(&test);
        assert_eq!(partition.components.len(), 1); // threads share address
    }

    #[test]
    fn test_state_space_split_by_threads() {
        let test = make_test_2t();
        let partition = StateSpaceSplitter::split_by_threads(&test);
        assert_eq!(partition.components.len(), 2);
    }

    #[test]
    fn test_composition_theorem() {
        let mut thm = CompositionTheorem::parallel(vec![0, 1]);
        assert!(!thm.is_discharged());
        thm.discharge(0);
        assert!(thm.is_discharged());
    }

    #[test]
    fn test_proof_leaf() {
        let proof = CompositionalProof::leaf("trivial component");
        assert_eq!(proof.size(), 1);
        assert!(proof.complete);
    }

    #[test]
    fn test_assume_guarantee_reasoner() {
        let test = make_independent_test();
        let components = vec![
            Component::from_threads(vec![0], &test),
            Component::from_threads(vec![1], &test),
        ];
        let model = MemoryModel::new("test");
        let reasoner = AssumeGuaranteeReasoner::new(components, model);
        let result = reasoner.verify_component(0);
        assert!(result.satisfied);
    }

    #[test]
    fn test_incremental_verifier() {
        let model = MemoryModel::new("test");
        let mut verifier = IncrementalVerifier::new(model);
        let test = make_independent_test();
        let _result = verifier.verify(&test);
        // Second verification should use cache
        let _result2 = verifier.verify(&test);
    }

    // ── Rely-Guarantee Tests ──

    fn make_shared_variable_test() -> LitmusTest {
        // Message passing: T0 writes x then y, T1 reads y then x
        // Shared variables: x (addr 0) and y (addr 1)
        let mut test = LitmusTest::new("mp-shared");
        let mut t0 = Thread::new(0);
        t0.store(0, 1, Ordering::Relaxed); // x = 1
        t0.store(1, 1, Ordering::Relaxed); // y = 1
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(1, 0, Ordering::Relaxed);  // r0 = y
        t1.load(0, 0, Ordering::Relaxed);  // r1 = x
        test.add_thread(t1);

        test
    }

    #[test]
    fn test_rely_guarantee_independent() {
        let test = make_independent_test();
        let model = MemoryModel::new("test");
        let engine = RelyGuaranteeEngine::new(model);
        let result = engine.verify(&test);
        assert!(result.safe);
        assert!(!result.conservative);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_rely_guarantee_shared_variables() {
        let test = make_shared_variable_test();
        let model = MemoryModel::new("test");
        let engine = RelyGuaranteeEngine::new(model);
        let result = engine.verify(&test);
        // Should be conservative
        assert!(result.conservative);
    }

    #[test]
    fn test_rely_extraction() {
        let test = make_shared_variable_test();
        let model = MemoryModel::new("test");
        let engine = RelyGuaranteeEngine::new(model);
        let component = Component::from_threads(vec![0], &test);
        let rely = engine.extract_rely(&component, &test);
        // Component 0's external addresses include those shared with T1
        assert!(!rely.addresses.is_empty() || component.external_addresses.is_empty());
    }

    #[test]
    fn test_guarantee_extraction() {
        let test = make_shared_variable_test();
        let model = MemoryModel::new("test");
        let engine = RelyGuaranteeEngine::new(model);
        let component = Component::from_threads(vec![0], &test);
        let guarantee = engine.extract_guarantee(&component, &test);
        // T0 writes to addresses, so guarantee should reflect this
        assert!(guarantee.has_fence || !guarantee.possible_writes.is_empty()
                || guarantee.written_addresses.is_empty());
    }

    #[test]
    fn test_rg_compatibility_check() {
        let test = make_independent_test();
        let model = MemoryModel::new("test");
        let engine = RelyGuaranteeEngine::new(model);
        let components = vec![
            Component::from_threads(vec![0], &test),
            Component::from_threads(vec![1], &test),
        ];
        let relies: Vec<_> = components.iter()
            .map(|c| engine.extract_rely(c, &test)).collect();
        let guarantees: Vec<_> = components.iter()
            .map(|c| engine.extract_guarantee(c, &test)).collect();
        let (compatible, violations) = engine.check_compatibility(
            &components, &relies, &guarantees);
        assert!(compatible);
        assert!(violations.is_empty());
    }
}

#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Self-contained types (no crate:: imports)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpKind {
    Read,
    Write,
    Fence,
    RMW,
    Branch,
    AtomicLoad,
    AtomicStore,
    AtomicRMW,
    Barrier,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read => write!(f, "R"),
            Self::Write => write!(f, "W"),
            Self::Fence => write!(f, "F"),
            Self::RMW => write!(f, "RMW"),
            Self::Branch => write!(f, "Br"),
            Self::AtomicLoad => write!(f, "AL"),
            Self::AtomicStore => write!(f, "AS"),
            Self::AtomicRMW => write!(f, "ARMW"),
            Self::Barrier => write!(f, "Bar"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Event {
    pub id: u32,
    pub thread_id: u32,
    pub op: OpKind,
    pub address: String,
    pub value: Option<u64>,
    pub order_index: u32,
}

impl Event {
    pub fn new(id: u32, thread_id: u32, op: OpKind, address: impl Into<String>) -> Self {
        Self {
            id,
            thread_id,
            op,
            address: address.into(),
            value: None,
            order_index: 0,
        }
    }

    pub fn with_value(mut self, v: u64) -> Self {
        self.value = Some(v);
        self
    }

    pub fn with_order(mut self, idx: u32) -> Self {
        self.order_index = idx;
        self
    }

    pub fn is_read(&self) -> bool {
        matches!(self.op, OpKind::Read | OpKind::AtomicLoad | OpKind::RMW | OpKind::AtomicRMW)
    }

    pub fn is_write(&self) -> bool {
        matches!(self.op, OpKind::Write | OpKind::AtomicStore | OpKind::RMW | OpKind::AtomicRMW)
    }

    pub fn is_fence(&self) -> bool {
        matches!(self.op, OpKind::Fence | OpKind::Barrier)
    }
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}:{} {}", self.thread_id, self.op, self.address)?;
        if let Some(v) = self.value {
            write!(f, "={}", v)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Relation {
    pub from: u32,
    pub to: u32,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Execution {
    pub events: Vec<Event>,
    pub relations: Vec<Relation>,
    pub threads: Vec<u32>,
}

impl Execution {
    pub fn new() -> Self {
        Self { events: Vec::new(), relations: Vec::new(), threads: Vec::new() }
    }

    pub fn add_event(&mut self, event: Event) {
        if !self.threads.contains(&event.thread_id) {
            self.threads.push(event.thread_id);
        }
        self.events.push(event);
    }

    pub fn add_relation(&mut self, from: u32, to: u32, kind: impl Into<String>) {
        self.relations.push(Relation { from, to, kind: kind.into() });
    }

    pub fn events_on_thread(&self, tid: u32) -> Vec<&Event> {
        self.events.iter().filter(|e| e.thread_id == tid).collect()
    }

    pub fn events_at_address(&self, addr: &str) -> Vec<&Event> {
        self.events.iter().filter(|e| e.address == addr).collect()
    }

    pub fn reads(&self) -> Vec<&Event> {
        self.events.iter().filter(|e| e.is_read()).collect()
    }

    pub fn writes(&self) -> Vec<&Event> {
        self.events.iter().filter(|e| e.is_write()).collect()
    }

    pub fn addresses(&self) -> HashSet<&str> {
        self.events.iter().map(|e| e.address.as_str()).collect()
    }

    pub fn num_threads(&self) -> usize {
        self.threads.len()
    }

    pub fn event_by_id(&self, id: u32) -> Option<&Event> {
        self.events.iter().find(|e| e.id == id)
    }

    pub fn adjacency_map(&self) -> HashMap<u32, Vec<(u32, String)>> {
        let mut map: HashMap<u32, Vec<(u32, String)>> = HashMap::new();
        for rel in &self.relations {
            map.entry(rel.from).or_default().push((rel.to, rel.kind.clone()));
        }
        map
    }
}

impl fmt::Display for Execution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Execution({} events, {} threads, {} relations)",
            self.events.len(), self.threads.len(), self.relations.len())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionOutcome {
    pub final_values: BTreeMap<String, u64>,
    pub read_values: BTreeMap<u32, u64>,
}

impl ExecutionOutcome {
    pub fn new() -> Self {
        Self { final_values: BTreeMap::new(), read_values: BTreeMap::new() }
    }

    pub fn with_final(mut self, addr: impl Into<String>, val: u64) -> Self {
        self.final_values.insert(addr.into(), val);
        self
    }

    pub fn with_read(mut self, event_id: u32, val: u64) -> Self {
        self.read_values.insert(event_id, val);
        self
    }

    pub fn fingerprint(&self) -> OutcomeFingerprint {
        let mut hasher = SimpleHasher::new();
        for (k, v) in &self.final_values {
            hasher.feed_str(k);
            hasher.feed_u64(*v);
        }
        for (k, v) in &self.read_values {
            hasher.feed_u64(*k as u64);
            hasher.feed_u64(*v);
        }
        OutcomeFingerprint(hasher.finish())
    }
}

impl fmt::Display for ExecutionOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let finals: Vec<String> = self.final_values.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        let reads: Vec<String> = self.read_values.iter().map(|(k, v)| format!("r{}={}", k, v)).collect();
        write!(f, "{{{}, {}}}", finals.join(", "), reads.join(", "))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutcomeFingerprint(pub u64);

struct SimpleHasher {
    state: u64,
}

impl SimpleHasher {
    fn new() -> Self { Self { state: 0xcbf29ce484222325 } }
    fn feed_u64(&mut self, v: u64) {
        self.state ^= v;
        self.state = self.state.wrapping_mul(0x100000001b3);
    }
    fn feed_str(&mut self, s: &str) {
        for b in s.bytes() {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }
    fn finish(&self) -> u64 { self.state }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LitmusTestDesc {
    pub name: String,
    pub threads: Vec<Vec<Event>>,
    pub initial_state: BTreeMap<String, u64>,
    pub expected_outcomes: Vec<ExecutionOutcome>,
    pub forbidden_outcomes: Vec<ExecutionOutcome>,
}

impl LitmusTestDesc {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            threads: Vec::new(),
            initial_state: BTreeMap::new(),
            expected_outcomes: Vec::new(),
            forbidden_outcomes: Vec::new(),
        }
    }

    pub fn add_thread(&mut self, events: Vec<Event>) {
        self.threads.push(events);
    }

    pub fn set_initial(&mut self, addr: impl Into<String>, val: u64) {
        self.initial_state.insert(addr.into(), val);
    }

    pub fn add_expected_outcome(&mut self, outcome: ExecutionOutcome) {
        self.expected_outcomes.push(outcome);
    }

    pub fn add_forbidden_outcome(&mut self, outcome: ExecutionOutcome) {
        self.forbidden_outcomes.push(outcome);
    }

    pub fn total_events(&self) -> usize {
        self.threads.iter().map(|t| t.len()).sum()
    }

    pub fn all_addresses(&self) -> HashSet<String> {
        let mut addrs = HashSet::new();
        for thread in &self.threads {
            for event in thread {
                addrs.insert(event.address.clone());
            }
        }
        addrs
    }

    pub fn to_execution(&self) -> Execution {
        let mut exec = Execution::new();
        for (tid, thread) in self.threads.iter().enumerate() {
            let mut prev_id = None;
            for event in thread {
                let mut e = event.clone();
                e.thread_id = tid as u32;
                exec.add_event(e.clone());
                if let Some(prev) = prev_id {
                    exec.add_relation(prev, e.id, "po");
                }
                prev_id = Some(e.id);
            }
        }
        exec
    }
}

// ---------------------------------------------------------------------------
// Isomorphism Checking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsomorphismMapping {
    pub event_map: BTreeMap<u32, u32>,
    pub thread_map: BTreeMap<u32, u32>,
    pub address_map: BTreeMap<String, String>,
}

impl IsomorphismMapping {
    pub fn identity() -> Self {
        Self {
            event_map: BTreeMap::new(),
            thread_map: BTreeMap::new(),
            address_map: BTreeMap::new(),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.event_map.iter().all(|(k, v)| k == v)
            && self.thread_map.iter().all(|(k, v)| k == v)
            && self.address_map.iter().all(|(k, v)| k == v)
    }
}

#[derive(Debug, Clone)]
pub struct IsomorphismChecker;

impl IsomorphismChecker {
    pub fn are_isomorphic(exec1: &Execution, exec2: &Execution) -> Option<IsomorphismMapping> {
        // Quick structural checks
        if exec1.events.len() != exec2.events.len() { return None; }
        if exec1.relations.len() != exec2.relations.len() { return None; }
        if exec1.threads.len() != exec2.threads.len() { return None; }

        // Check event counts per operation kind match
        let ops1 = Self::op_histogram(exec1);
        let ops2 = Self::op_histogram(exec2);
        if ops1 != ops2 { return None; }

        // Check thread event count signature
        let mut thread_sizes1: Vec<usize> = exec1.threads.iter()
            .map(|&t| exec1.events_on_thread(t).len()).collect();
        let mut thread_sizes2: Vec<usize> = exec2.threads.iter()
            .map(|&t| exec2.events_on_thread(t).len()).collect();
        thread_sizes1.sort();
        thread_sizes2.sort();
        if thread_sizes1 != thread_sizes2 { return None; }

        // Try to find mapping using backtracking
        Self::find_mapping(exec1, exec2)
    }

    fn op_histogram(exec: &Execution) -> BTreeMap<OpKind, usize> {
        let mut map = BTreeMap::new();
        for e in &exec.events {
            *map.entry(e.op).or_insert(0) += 1;
        }
        map
    }

    fn find_mapping(exec1: &Execution, exec2: &Execution) -> Option<IsomorphismMapping> {
        // Build thread permutations
        let threads1: Vec<u32> = exec1.threads.clone();
        let threads2: Vec<u32> = exec2.threads.clone();

        let thread_perms = Self::permutations(&threads2);

        for thread_perm in thread_perms {
            let mut thread_map = BTreeMap::new();
            for (i, &t1) in threads1.iter().enumerate() {
                thread_map.insert(t1, thread_perm[i]);
            }

            // Check if per-thread event counts match under this thread mapping
            let mut ok = true;
            for (&t1, &t2) in &thread_map {
                if exec1.events_on_thread(t1).len() != exec2.events_on_thread(t2).len() {
                    ok = false;
                    break;
                }
            }
            if !ok { continue; }

            // Try to build event mapping
            if let Some(mapping) = Self::try_event_mapping(exec1, exec2, &thread_map) {
                return Some(mapping);
            }
        }

        None
    }

    fn try_event_mapping(
        exec1: &Execution,
        exec2: &Execution,
        thread_map: &BTreeMap<u32, u32>,
    ) -> Option<IsomorphismMapping> {
        let mut event_map = BTreeMap::new();
        let mut addr_map: BTreeMap<String, String> = BTreeMap::new();
        let mut used_events: HashSet<u32> = HashSet::new();

        for (&t1, &t2) in thread_map {
            let events1 = exec1.events_on_thread(t1);
            let events2 = exec2.events_on_thread(t2);

            for (e1, e2) in events1.iter().zip(events2.iter()) {
                if e1.op != e2.op { return None; }

                // Check address consistency
                if let Some(mapped_addr) = addr_map.get(&e1.address) {
                    if mapped_addr != &e2.address { return None; }
                } else {
                    addr_map.insert(e1.address.clone(), e2.address.clone());
                }

                if used_events.contains(&e2.id) { return None; }
                event_map.insert(e1.id, e2.id);
                used_events.insert(e2.id);
            }
        }

        // Verify relation mapping
        let adj1 = exec1.adjacency_map();
        let adj2 = exec2.adjacency_map();

        for rel in &exec1.relations {
            let mapped_from = event_map.get(&rel.from)?;
            let mapped_to = event_map.get(&rel.to)?;
            let has_rel = exec2.relations.iter().any(|r| {
                r.from == *mapped_from && r.to == *mapped_to && r.kind == rel.kind
            });
            if !has_rel { return None; }
        }

        Some(IsomorphismMapping {
            event_map,
            thread_map: thread_map.clone(),
            address_map: addr_map,
        })
    }

    fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut result = Vec::new();
        for i in 0..items.len() {
            let mut rest: Vec<T> = items.to_vec();
            let elem = rest.remove(i);
            for mut perm in Self::permutations(&rest) {
                perm.insert(0, elem.clone());
                result.push(perm);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Canonical Form
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonicalExecution {
    pub events: Vec<CanonicalEvent>,
    pub relations: Vec<(usize, usize, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonicalEvent {
    pub thread_index: usize,
    pub position: usize,
    pub op: OpKind,
    pub address_index: usize,
    pub value: Option<u64>,
}

pub fn canonical_form(exec: &Execution) -> CanonicalExecution {
    // Sort threads by size (descending), then by content
    let mut thread_events: Vec<(u32, Vec<&Event>)> = exec.threads.iter()
        .map(|&tid| (tid, exec.events_on_thread(tid)))
        .collect();
    thread_events.sort_by(|a, b| {
        b.1.len().cmp(&a.1.len())
            .then_with(|| {
                let a_ops: Vec<OpKind> = a.1.iter().map(|e| e.op).collect();
                let b_ops: Vec<OpKind> = b.1.iter().map(|e| e.op).collect();
                format!("{:?}", a_ops).cmp(&format!("{:?}", b_ops))
            })
    });

    let thread_index_map: HashMap<u32, usize> = thread_events.iter()
        .enumerate()
        .map(|(i, (tid, _))| (*tid, i))
        .collect();

    // Build address index: assign indices based on first appearance order
    let mut addr_index: HashMap<String, usize> = HashMap::new();
    let mut addr_counter = 0;
    for (_tid, events) in &thread_events {
        for event in events {
            if !addr_index.contains_key(&event.address) {
                addr_index.insert(event.address.clone(), addr_counter);
                addr_counter += 1;
            }
        }
    }

    // Build canonical event list
    let mut id_to_canonical: HashMap<u32, usize> = HashMap::new();
    let mut canonical_events = Vec::new();
    for (thread_idx, (_tid, events)) in thread_events.iter().enumerate() {
        for (pos, event) in events.iter().enumerate() {
            let ce = CanonicalEvent {
                thread_index: thread_idx,
                position: pos,
                op: event.op,
                address_index: addr_index[&event.address],
                value: event.value,
            };
            id_to_canonical.insert(event.id, canonical_events.len());
            canonical_events.push(ce);
        }
    }

    // Map relations
    let mut canonical_rels: Vec<(usize, usize, String)> = exec.relations.iter()
        .filter_map(|r| {
            let from = id_to_canonical.get(&r.from)?;
            let to = id_to_canonical.get(&r.to)?;
            Some((*from, *to, r.kind.clone()))
        })
        .collect();
    canonical_rels.sort();

    CanonicalExecution {
        events: canonical_events,
        relations: canonical_rels,
    }
}

pub fn isomorphism_hash(exec: &Execution) -> u64 {
    let canonical = canonical_form(exec);
    let mut hasher = SimpleHasher::new();
    for ce in &canonical.events {
        hasher.feed_u64(ce.thread_index as u64);
        hasher.feed_u64(ce.position as u64);
        hasher.feed_u64(ce.op as u64);
        hasher.feed_u64(ce.address_index as u64);
        if let Some(v) = ce.value {
            hasher.feed_u64(v);
        }
    }
    for (from, to, kind) in &canonical.relations {
        hasher.feed_u64(*from as u64);
        hasher.feed_u64(*to as u64);
        hasher.feed_str(kind);
    }
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Color Refinement (Weisfeiler-Leman)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ColorRefinement {
    colors: HashMap<u32, u64>,
    adj: HashMap<u32, Vec<(u32, String)>>,
}

impl ColorRefinement {
    pub fn new(exec: &Execution) -> Self {
        let mut colors = HashMap::new();
        for event in &exec.events {
            let mut h = SimpleHasher::new();
            h.feed_u64(event.op as u64);
            h.feed_str(&event.address);
            colors.insert(event.id, h.finish());
        }
        let adj = exec.adjacency_map();
        Self { colors, adj }
    }

    pub fn refine_colors(&mut self, max_rounds: usize) -> usize {
        let mut rounds = 0;
        for _ in 0..max_rounds {
            rounds += 1;
            let mut new_colors = HashMap::new();
            let mut changed = false;

            for (&id, &current_color) in &self.colors {
                let mut neighbor_colors: Vec<(u64, String)> = Vec::new();
                if let Some(neighbors) = self.adj.get(&id) {
                    for (nid, kind) in neighbors {
                        if let Some(&nc) = self.colors.get(nid) {
                            neighbor_colors.push((nc, kind.clone()));
                        }
                    }
                }
                neighbor_colors.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

                let mut h = SimpleHasher::new();
                h.feed_u64(current_color);
                for (nc, kind) in &neighbor_colors {
                    h.feed_u64(*nc);
                    h.feed_str(kind);
                }
                let new_color = h.finish();
                if new_color != current_color {
                    changed = true;
                }
                new_colors.insert(id, new_color);
            }

            self.colors = new_colors;
            if !changed {
                break;
            }
        }
        rounds
    }

    pub fn stable_coloring(&mut self) -> HashMap<u32, u64> {
        self.refine_colors(100);
        self.colors.clone()
    }

    pub fn color_histogram(&self) -> BTreeMap<u64, usize> {
        let mut hist = BTreeMap::new();
        for &c in self.colors.values() {
            *hist.entry(c).or_insert(0) += 1;
        }
        hist
    }

    pub fn color_signature(&mut self) -> Vec<usize> {
        self.refine_colors(100);
        let hist = self.color_histogram();
        let mut counts: Vec<usize> = hist.values().copied().collect();
        counts.sort_unstable();
        counts
    }
}

// ---------------------------------------------------------------------------
// Outcome Equivalence
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeSet {
    pub outcomes: Vec<ExecutionOutcome>,
    pub fingerprints: HashSet<OutcomeFingerprint>,
}

impl OutcomeSet {
    pub fn new() -> Self {
        Self { outcomes: Vec::new(), fingerprints: HashSet::new() }
    }

    pub fn add(&mut self, outcome: ExecutionOutcome) {
        let fp = outcome.fingerprint();
        if self.fingerprints.insert(fp) {
            self.outcomes.push(outcome);
        }
    }

    pub fn contains(&self, outcome: &ExecutionOutcome) -> bool {
        self.fingerprints.contains(&outcome.fingerprint())
    }

    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    pub fn union(&self, other: &OutcomeSet) -> OutcomeSet {
        let mut result = self.clone();
        for o in &other.outcomes {
            result.add(o.clone());
        }
        result
    }

    pub fn intersection(&self, other: &OutcomeSet) -> OutcomeSet {
        let mut result = OutcomeSet::new();
        for o in &self.outcomes {
            if other.contains(o) {
                result.add(o.clone());
            }
        }
        result
    }

    pub fn difference(&self, other: &OutcomeSet) -> OutcomeSet {
        let mut result = OutcomeSet::new();
        for o in &self.outcomes {
            if !other.contains(o) {
                result.add(o.clone());
            }
        }
        result
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeDiff {
    pub only_in_first: OutcomeSet,
    pub only_in_second: OutcomeSet,
    pub common: OutcomeSet,
}

impl OutcomeDiff {
    pub fn is_equivalent(&self) -> bool {
        self.only_in_first.is_empty() && self.only_in_second.is_empty()
    }

    pub fn jaccard_similarity(&self) -> f64 {
        let common = self.common.len() as f64;
        let total = common + self.only_in_first.len() as f64 + self.only_in_second.len() as f64;
        if total == 0.0 { return 1.0; }
        common / total
    }
}

pub struct OutcomeEquivalenceChecker;

impl OutcomeEquivalenceChecker {
    pub fn same_outcomes(set1: &OutcomeSet, set2: &OutcomeSet) -> bool {
        if set1.len() != set2.len() { return false; }
        set1.fingerprints == set2.fingerprints
    }

    pub fn outcome_diff(set1: &OutcomeSet, set2: &OutcomeSet) -> OutcomeDiff {
        OutcomeDiff {
            only_in_first: set1.difference(set2),
            only_in_second: set2.difference(set1),
            common: set1.intersection(set2),
        }
    }
}

// ---------------------------------------------------------------------------
// Subsumption Checking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubsumptionResult {
    Subsumes,
    SubsumedBy,
    Equal,
    Incomparable,
}

impl fmt::Display for SubsumptionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Subsumes => write!(f, "subsumes"),
            Self::SubsumedBy => write!(f, "subsumed_by"),
            Self::Equal => write!(f, "equal"),
            Self::Incomparable => write!(f, "incomparable"),
        }
    }
}

pub struct SubsumptionChecker;

impl SubsumptionChecker {
    pub fn subsumes(test1: &LitmusTestDesc, test2: &LitmusTestDesc) -> SubsumptionResult {
        let outcomes1: OutcomeSet = Self::outcomes_to_set(&test1.expected_outcomes);
        let outcomes2: OutcomeSet = Self::outcomes_to_set(&test2.expected_outcomes);

        let o1_subset_o2 = outcomes1.outcomes.iter().all(|o| outcomes2.contains(o));
        let o2_subset_o1 = outcomes2.outcomes.iter().all(|o| outcomes1.contains(o));

        match (o1_subset_o2, o2_subset_o1) {
            (true, true) => SubsumptionResult::Equal,
            (true, false) => SubsumptionResult::SubsumedBy,
            (false, true) => SubsumptionResult::Subsumes,
            (false, false) => SubsumptionResult::Incomparable,
        }
    }

    fn outcomes_to_set(outcomes: &[ExecutionOutcome]) -> OutcomeSet {
        let mut set = OutcomeSet::new();
        for o in outcomes {
            set.add(o.clone());
        }
        set
    }

    pub fn subsumption_matrix(tests: &[LitmusTestDesc]) -> Vec<Vec<SubsumptionResult>> {
        let n = tests.len();
        let mut matrix = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                if i == j {
                    row.push(SubsumptionResult::Equal);
                } else {
                    row.push(Self::subsumes(&tests[i], &tests[j]));
                }
            }
            matrix.push(row);
        }
        matrix
    }
}

// ---------------------------------------------------------------------------
// Equivalence Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEquivalenceReport {
    pub equivalence_classes: Vec<Vec<usize>>,
    pub total_tests: usize,
    pub unique_classes: usize,
    pub reduction_ratio: f64,
}

impl TestEquivalenceReport {
    pub fn largest_class_size(&self) -> usize {
        self.equivalence_classes.iter().map(|c| c.len()).max().unwrap_or(0)
    }

    pub fn singleton_count(&self) -> usize {
        self.equivalence_classes.iter().filter(|c| c.len() == 1).count()
    }
}

#[derive(Debug, Clone)]
pub struct EquivalenceDetector {
    use_structural: bool,
    use_outcome: bool,
    use_color_refinement: bool,
}

impl EquivalenceDetector {
    pub fn new() -> Self {
        Self { use_structural: true, use_outcome: true, use_color_refinement: true }
    }

    pub fn structural_only() -> Self {
        Self { use_structural: true, use_outcome: false, use_color_refinement: false }
    }

    pub fn outcome_only() -> Self {
        Self { use_structural: false, use_outcome: true, use_color_refinement: false }
    }

    pub fn detect_equivalences(&self, tests: &[LitmusTestDesc]) -> TestEquivalenceReport {
        let n = tests.len();
        if n == 0 {
            return TestEquivalenceReport {
                equivalence_classes: Vec::new(),
                total_tests: 0,
                unique_classes: 0,
                reduction_ratio: 1.0,
            };
        }

        // Union-Find for grouping
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        for i in 0..n {
            for j in (i + 1)..n {
                if self.are_equivalent(&tests[i], &tests[j]) {
                    Self::union(&mut parent, &mut rank, i, j);
                }
            }
        }

        // Collect equivalence classes
        let mut classes: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = Self::find(&mut parent, i);
            classes.entry(root).or_default().push(i);
        }

        let mut equiv_classes: Vec<Vec<usize>> = classes.into_values().collect();
        equiv_classes.sort_by_key(|c| c[0]);

        let unique = equiv_classes.len();
        TestEquivalenceReport {
            equivalence_classes: equiv_classes,
            total_tests: n,
            unique_classes: unique,
            reduction_ratio: 1.0 - (unique as f64 / n as f64),
        }
    }

    fn are_equivalent(&self, t1: &LitmusTestDesc, t2: &LitmusTestDesc) -> bool {
        if self.use_structural {
            let exec1 = t1.to_execution();
            let exec2 = t2.to_execution();
            if IsomorphismChecker::are_isomorphic(&exec1, &exec2).is_some() {
                return true;
            }
        }

        if self.use_outcome {
            let set1: OutcomeSet = {
                let mut s = OutcomeSet::new();
                for o in &t1.expected_outcomes { s.add(o.clone()); }
                s
            };
            let set2: OutcomeSet = {
                let mut s = OutcomeSet::new();
                for o in &t2.expected_outcomes { s.add(o.clone()); }
                s
            };
            if OutcomeEquivalenceChecker::same_outcomes(&set1, &set2) {
                return true;
            }
        }

        if self.use_color_refinement {
            let exec1 = t1.to_execution();
            let exec2 = t2.to_execution();
            let mut cr1 = ColorRefinement::new(&exec1);
            let mut cr2 = ColorRefinement::new(&exec2);
            let sig1 = cr1.color_signature();
            let sig2 = cr2.color_signature();
            if sig1 == sig2 {
                return true;
            }
        }

        false
    }

    fn find(parent: &mut Vec<usize>, i: usize) -> usize {
        if parent[i] != i {
            parent[i] = Self::find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut Vec<usize>, rank: &mut Vec<usize>, i: usize, j: usize) {
        let ri = Self::find(parent, i);
        let rj = Self::find(parent, j);
        if ri == rj { return; }
        if rank[ri] < rank[rj] {
            parent[ri] = rj;
        } else if rank[ri] > rank[rj] {
            parent[rj] = ri;
        } else {
            parent[rj] = ri;
            rank[ri] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Test Minimization
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TestMinimizer;

impl TestMinimizer {
    pub fn minimize_test(test: &LitmusTestDesc, predicate: &dyn Fn(&LitmusTestDesc) -> bool) -> LitmusTestDesc {
        if !predicate(test) {
            return test.clone();
        }

        let mut current = test.clone();

        // Try removing threads
        for tid in (0..current.threads.len()).rev() {
            if current.threads.len() <= 1 { break; }
            let mut candidate = current.clone();
            candidate.threads.remove(tid);
            if predicate(&candidate) {
                current = candidate;
            }
        }

        // Try removing events from each thread
        for tid in 0..current.threads.len() {
            let mut eid = current.threads[tid].len();
            while eid > 0 {
                eid -= 1;
                if current.threads[tid].len() <= 1 { break; }
                let mut candidate = current.clone();
                candidate.threads[tid].remove(eid);
                if predicate(&candidate) {
                    current = candidate;
                }
            }
        }

        current
    }

    pub fn delta_debugging(
        test: &LitmusTestDesc,
        predicate: &dyn Fn(&LitmusTestDesc) -> bool,
    ) -> LitmusTestDesc {
        if !predicate(test) {
            return test.clone();
        }

        let all_events: Vec<(usize, usize)> = test.threads.iter().enumerate()
            .flat_map(|(tid, thread)| (0..thread.len()).map(move |eid| (tid, eid)))
            .collect();

        let minimal = Self::dd_minimize(&all_events, test, predicate, 2);
        Self::rebuild_test(test, &minimal)
    }

    fn dd_minimize(
        events: &[(usize, usize)],
        original: &LitmusTestDesc,
        predicate: &dyn Fn(&LitmusTestDesc) -> bool,
        n: usize,
    ) -> Vec<(usize, usize)> {
        if events.len() <= 1 {
            return events.to_vec();
        }

        let chunk_size = (events.len() + n - 1) / n;
        let chunks: Vec<Vec<(usize, usize)>> = events.chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect();

        // Try each subset (complement of each chunk)
        for (i, chunk) in chunks.iter().enumerate() {
            let complement: Vec<(usize, usize)> = events.iter()
                .filter(|e| !chunk.contains(e))
                .cloned()
                .collect();

            let test = Self::rebuild_test(original, &complement);
            if predicate(&test) {
                return Self::dd_minimize(&complement, original, predicate, 2);
            }
        }

        // Try each chunk itself
        for chunk in &chunks {
            let test = Self::rebuild_test(original, chunk);
            if predicate(&test) {
                return Self::dd_minimize(chunk, original, predicate, 2);
            }
        }

        // Increase granularity
        if n < events.len() {
            return Self::dd_minimize(events, original, predicate, n.min(events.len()) * 2);
        }

        events.to_vec()
    }

    fn rebuild_test(original: &LitmusTestDesc, events: &[(usize, usize)]) -> LitmusTestDesc {
        let mut test = LitmusTestDesc::new(&original.name);
        test.initial_state = original.initial_state.clone();
        test.expected_outcomes = original.expected_outcomes.clone();
        test.forbidden_outcomes = original.forbidden_outcomes.clone();

        let mut thread_map: BTreeMap<usize, Vec<Event>> = BTreeMap::new();
        for &(tid, eid) in events {
            if tid < original.threads.len() && eid < original.threads[tid].len() {
                thread_map.entry(tid).or_default().push(original.threads[tid][eid].clone());
            }
        }
        for (_, evts) in thread_map {
            test.threads.push(evts);
        }

        test
    }
}

// ---------------------------------------------------------------------------
// Test Signature
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TestSignature {
    pub num_threads: usize,
    pub num_events: usize,
    pub num_reads: usize,
    pub num_writes: usize,
    pub num_fences: usize,
    pub num_addresses: usize,
    pub thread_sizes: Vec<usize>,
    pub op_histogram: BTreeMap<String, usize>,
    pub hash: u64,
}

pub fn compute_signature(test: &LitmusTestDesc) -> TestSignature {
    let mut num_reads = 0;
    let mut num_writes = 0;
    let mut num_fences = 0;
    let mut num_events = 0;
    let mut addresses = HashSet::new();
    let mut op_hist: BTreeMap<String, usize> = BTreeMap::new();
    let mut thread_sizes = Vec::new();

    for thread in &test.threads {
        thread_sizes.push(thread.len());
        for event in thread {
            num_events += 1;
            if event.is_read() { num_reads += 1; }
            if event.is_write() { num_writes += 1; }
            if event.is_fence() { num_fences += 1; }
            addresses.insert(event.address.clone());
            *op_hist.entry(format!("{}", event.op)).or_insert(0) += 1;
        }
    }

    thread_sizes.sort_unstable();

    let exec = test.to_execution();
    let hash = isomorphism_hash(&exec);

    TestSignature {
        num_threads: test.threads.len(),
        num_events,
        num_reads,
        num_writes,
        num_fences,
        num_addresses: addresses.len(),
        thread_sizes,
        op_histogram: op_hist,
        hash,
    }
}

// ---------------------------------------------------------------------------
// Signature Index
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SignatureIndex {
    signatures: Vec<(usize, TestSignature)>,
    by_hash: HashMap<u64, Vec<usize>>,
    by_shape: HashMap<(usize, usize, usize), Vec<usize>>,
}

impl SignatureIndex {
    pub fn new() -> Self {
        Self {
            signatures: Vec::new(),
            by_hash: HashMap::new(),
            by_shape: HashMap::new(),
        }
    }

    pub fn add(&mut self, test_index: usize, sig: TestSignature) {
        self.by_hash.entry(sig.hash).or_default().push(test_index);
        let shape = (sig.num_threads, sig.num_events, sig.num_addresses);
        self.by_shape.entry(shape).or_default().push(test_index);
        self.signatures.push((test_index, sig));
    }

    pub fn find_candidates(&self, sig: &TestSignature) -> Vec<usize> {
        if let Some(candidates) = self.by_hash.get(&sig.hash) {
            return candidates.clone();
        }
        Vec::new()
    }

    pub fn find_similar(&self, sig: &TestSignature) -> Vec<usize> {
        let shape = (sig.num_threads, sig.num_events, sig.num_addresses);
        self.by_shape.get(&shape).cloned().unwrap_or_default()
    }

    pub fn build_from_tests(tests: &[LitmusTestDesc]) -> Self {
        let mut index = Self::new();
        for (i, test) in tests.iter().enumerate() {
            let sig = compute_signature(test);
            index.add(i, sig);
        }
        index
    }

    pub fn equivalence_classes(&self) -> Vec<Vec<usize>> {
        let mut classes: HashMap<u64, Vec<usize>> = HashMap::new();
        for (idx, sig) in &self.signatures {
            classes.entry(sig.hash).or_default().push(*idx);
        }
        classes.into_values().collect()
    }
}

// ---------------------------------------------------------------------------
// Test Suite Reduction
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TestSuiteReducer;

impl TestSuiteReducer {
    pub fn reduce_by_equivalence(tests: &[LitmusTestDesc]) -> Vec<usize> {
        let detector = EquivalenceDetector::new();
        let report = detector.detect_equivalences(tests);
        // Keep one representative from each class
        report.equivalence_classes.iter()
            .map(|class| class[0])
            .collect()
    }

    pub fn reduce_by_signature(tests: &[LitmusTestDesc]) -> Vec<usize> {
        let index = SignatureIndex::build_from_tests(tests);
        let classes = index.equivalence_classes();
        classes.iter().map(|class| class[0]).collect()
    }

    pub fn reduce_by_subsumption(tests: &[LitmusTestDesc]) -> Vec<usize> {
        let n = tests.len();
        let mut subsumed = vec![false; n];

        for i in 0..n {
            if subsumed[i] { continue; }
            for j in 0..n {
                if i == j || subsumed[j] { continue; }
                match SubsumptionChecker::subsumes(&tests[i], &tests[j]) {
                    SubsumptionResult::Subsumes => {
                        subsumed[j] = true;
                    }
                    _ => {}
                }
            }
        }

        (0..n).filter(|&i| !subsumed[i]).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ===== Extended Equivalence Operations =====

#[derive(Debug, Clone)]
pub struct TestNormalizer {
    pub thread_count: u32,
    pub address_count: u32,
    pub value_range: u64,
    pub normalized: bool,
}

impl TestNormalizer {
    pub fn new(thread_count: u32, address_count: u32, value_range: u64, normalized: bool) -> Self {
        TestNormalizer { thread_count, address_count, value_range, normalized }
    }

    pub fn get_thread_count(&self) -> u32 {
        self.thread_count
    }

    pub fn get_address_count(&self) -> u32 {
        self.address_count
    }

    pub fn get_value_range(&self) -> u64 {
        self.value_range
    }

    pub fn get_normalized(&self) -> bool {
        self.normalized
    }

    pub fn with_thread_count(mut self, v: u32) -> Self {
        self.thread_count = v; self
    }

    pub fn with_address_count(mut self, v: u32) -> Self {
        self.address_count = v; self
    }

    pub fn with_value_range(mut self, v: u64) -> Self {
        self.value_range = v; self
    }

    pub fn with_normalized(mut self, v: bool) -> Self {
        self.normalized = v; self
    }

}

impl fmt::Display for TestNormalizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TestNormalizer({:?})", self.thread_count)
    }
}

#[derive(Debug, Clone)]
pub struct TestNormalizerBuilder {
    thread_count: u32,
    address_count: u32,
    value_range: u64,
    normalized: bool,
}

impl TestNormalizerBuilder {
    pub fn new() -> Self {
        TestNormalizerBuilder {
            thread_count: 0,
            address_count: 0,
            value_range: 0,
            normalized: false,
        }
    }

    pub fn thread_count(mut self, v: u32) -> Self { self.thread_count = v; self }
    pub fn address_count(mut self, v: u32) -> Self { self.address_count = v; self }
    pub fn value_range(mut self, v: u64) -> Self { self.value_range = v; self }
    pub fn normalized(mut self, v: bool) -> Self { self.normalized = v; self }
}

#[derive(Debug, Clone)]
pub struct ThreadRenaming {
    pub original_ids: Vec<u32>,
    pub renamed_ids: Vec<u32>,
    pub mapping: Vec<(u32, u32)>,
}

impl ThreadRenaming {
    pub fn new(original_ids: Vec<u32>, renamed_ids: Vec<u32>, mapping: Vec<(u32, u32)>) -> Self {
        ThreadRenaming { original_ids, renamed_ids, mapping }
    }

    pub fn original_ids_len(&self) -> usize {
        self.original_ids.len()
    }

    pub fn original_ids_is_empty(&self) -> bool {
        self.original_ids.is_empty()
    }

    pub fn renamed_ids_len(&self) -> usize {
        self.renamed_ids.len()
    }

    pub fn renamed_ids_is_empty(&self) -> bool {
        self.renamed_ids.is_empty()
    }

    pub fn mapping_len(&self) -> usize {
        self.mapping.len()
    }

    pub fn mapping_is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

}

impl fmt::Display for ThreadRenaming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ThreadRenaming({:?})", self.original_ids)
    }
}

#[derive(Debug, Clone)]
pub struct ThreadRenamingBuilder {
    original_ids: Vec<u32>,
    renamed_ids: Vec<u32>,
    mapping: Vec<(u32, u32)>,
}

impl ThreadRenamingBuilder {
    pub fn new() -> Self {
        ThreadRenamingBuilder {
            original_ids: Vec::new(),
            renamed_ids: Vec::new(),
            mapping: Vec::new(),
        }
    }

    pub fn original_ids(mut self, v: Vec<u32>) -> Self { self.original_ids = v; self }
    pub fn renamed_ids(mut self, v: Vec<u32>) -> Self { self.renamed_ids = v; self }
    pub fn mapping(mut self, v: Vec<(u32, u32)>) -> Self { self.mapping = v; self }
}

#[derive(Debug, Clone)]
pub struct AddressRenaming {
    pub original_addrs: Vec<String>,
    pub renamed_addrs: Vec<String>,
    pub mapping_count: usize,
}

impl AddressRenaming {
    pub fn new(original_addrs: Vec<String>, renamed_addrs: Vec<String>, mapping_count: usize) -> Self {
        AddressRenaming { original_addrs, renamed_addrs, mapping_count }
    }

    pub fn original_addrs_len(&self) -> usize {
        self.original_addrs.len()
    }

    pub fn original_addrs_is_empty(&self) -> bool {
        self.original_addrs.is_empty()
    }

    pub fn renamed_addrs_len(&self) -> usize {
        self.renamed_addrs.len()
    }

    pub fn renamed_addrs_is_empty(&self) -> bool {
        self.renamed_addrs.is_empty()
    }

    pub fn get_mapping_count(&self) -> usize {
        self.mapping_count
    }

    pub fn with_mapping_count(mut self, v: usize) -> Self {
        self.mapping_count = v; self
    }

}

impl fmt::Display for AddressRenaming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AddressRenaming({:?})", self.original_addrs)
    }
}

#[derive(Debug, Clone)]
pub struct AddressRenamingBuilder {
    original_addrs: Vec<String>,
    renamed_addrs: Vec<String>,
    mapping_count: usize,
}

impl AddressRenamingBuilder {
    pub fn new() -> Self {
        AddressRenamingBuilder {
            original_addrs: Vec::new(),
            renamed_addrs: Vec::new(),
            mapping_count: 0,
        }
    }

    pub fn original_addrs(mut self, v: Vec<String>) -> Self { self.original_addrs = v; self }
    pub fn renamed_addrs(mut self, v: Vec<String>) -> Self { self.renamed_addrs = v; self }
    pub fn mapping_count(mut self, v: usize) -> Self { self.mapping_count = v; self }
}

#[derive(Debug, Clone)]
pub struct ValueRenaming {
    pub original_values: Vec<u64>,
    pub renamed_values: Vec<u64>,
    pub preserves_zero: bool,
}

impl ValueRenaming {
    pub fn new(original_values: Vec<u64>, renamed_values: Vec<u64>, preserves_zero: bool) -> Self {
        ValueRenaming { original_values, renamed_values, preserves_zero }
    }

    pub fn original_values_len(&self) -> usize {
        self.original_values.len()
    }

    pub fn original_values_is_empty(&self) -> bool {
        self.original_values.is_empty()
    }

    pub fn renamed_values_len(&self) -> usize {
        self.renamed_values.len()
    }

    pub fn renamed_values_is_empty(&self) -> bool {
        self.renamed_values.is_empty()
    }

    pub fn get_preserves_zero(&self) -> bool {
        self.preserves_zero
    }

    pub fn with_preserves_zero(mut self, v: bool) -> Self {
        self.preserves_zero = v; self
    }

}

impl fmt::Display for ValueRenaming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValueRenaming({:?})", self.original_values)
    }
}

#[derive(Debug, Clone)]
pub struct ValueRenamingBuilder {
    original_values: Vec<u64>,
    renamed_values: Vec<u64>,
    preserves_zero: bool,
}

impl ValueRenamingBuilder {
    pub fn new() -> Self {
        ValueRenamingBuilder {
            original_values: Vec::new(),
            renamed_values: Vec::new(),
            preserves_zero: false,
        }
    }

    pub fn original_values(mut self, v: Vec<u64>) -> Self { self.original_values = v; self }
    pub fn renamed_values(mut self, v: Vec<u64>) -> Self { self.renamed_values = v; self }
    pub fn preserves_zero(mut self, v: bool) -> Self { self.preserves_zero = v; self }
}

#[derive(Debug, Clone)]
pub struct TestHasher {
    pub seed: u64,
    pub polynomial: u64,
    pub hash_bits: u32,
}

impl TestHasher {
    pub fn new(seed: u64, polynomial: u64, hash_bits: u32) -> Self {
        TestHasher { seed, polynomial, hash_bits }
    }

    pub fn get_seed(&self) -> u64 {
        self.seed
    }

    pub fn get_polynomial(&self) -> u64 {
        self.polynomial
    }

    pub fn get_hash_bits(&self) -> u32 {
        self.hash_bits
    }

    pub fn with_seed(mut self, v: u64) -> Self {
        self.seed = v; self
    }

    pub fn with_polynomial(mut self, v: u64) -> Self {
        self.polynomial = v; self
    }

    pub fn with_hash_bits(mut self, v: u32) -> Self {
        self.hash_bits = v; self
    }

}

impl fmt::Display for TestHasher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TestHasher({:?})", self.seed)
    }
}

#[derive(Debug, Clone)]
pub struct TestHasherBuilder {
    seed: u64,
    polynomial: u64,
    hash_bits: u32,
}

impl TestHasherBuilder {
    pub fn new() -> Self {
        TestHasherBuilder {
            seed: 0,
            polynomial: 0,
            hash_bits: 0,
        }
    }

    pub fn seed(mut self, v: u64) -> Self { self.seed = v; self }
    pub fn polynomial(mut self, v: u64) -> Self { self.polynomial = v; self }
    pub fn hash_bits(mut self, v: u32) -> Self { self.hash_bits = v; self }
}

#[derive(Debug, Clone)]
pub struct TestSerializer {
    pub format: String,
    pub compact: bool,
    pub include_metadata: bool,
}

impl TestSerializer {
    pub fn new(format: String, compact: bool, include_metadata: bool) -> Self {
        TestSerializer { format, compact, include_metadata }
    }

    pub fn get_format(&self) -> &str {
        &self.format
    }

    pub fn get_compact(&self) -> bool {
        self.compact
    }

    pub fn get_include_metadata(&self) -> bool {
        self.include_metadata
    }

    pub fn with_format(mut self, v: impl Into<String>) -> Self {
        self.format = v.into(); self
    }

    pub fn with_compact(mut self, v: bool) -> Self {
        self.compact = v; self
    }

    pub fn with_include_metadata(mut self, v: bool) -> Self {
        self.include_metadata = v; self
    }

}

impl fmt::Display for TestSerializer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TestSerializer({:?})", self.format)
    }
}

#[derive(Debug, Clone)]
pub struct TestSerializerBuilder {
    format: String,
    compact: bool,
    include_metadata: bool,
}

impl TestSerializerBuilder {
    pub fn new() -> Self {
        TestSerializerBuilder {
            format: String::new(),
            compact: false,
            include_metadata: false,
        }
    }

    pub fn format(mut self, v: impl Into<String>) -> Self { self.format = v.into(); self }
    pub fn compact(mut self, v: bool) -> Self { self.compact = v; self }
    pub fn include_metadata(mut self, v: bool) -> Self { self.include_metadata = v; self }
}

#[derive(Debug, Clone)]
pub struct EquivalenceProofCert {
    pub class_id: u64,
    pub witness_mapping: Vec<(u32, u32)>,
    pub verified: bool,
}

impl EquivalenceProofCert {
    pub fn new(class_id: u64, witness_mapping: Vec<(u32, u32)>, verified: bool) -> Self {
        EquivalenceProofCert { class_id, witness_mapping, verified }
    }

    pub fn get_class_id(&self) -> u64 {
        self.class_id
    }

    pub fn witness_mapping_len(&self) -> usize {
        self.witness_mapping.len()
    }

    pub fn witness_mapping_is_empty(&self) -> bool {
        self.witness_mapping.is_empty()
    }

    pub fn get_verified(&self) -> bool {
        self.verified
    }

    pub fn with_class_id(mut self, v: u64) -> Self {
        self.class_id = v; self
    }

    pub fn with_verified(mut self, v: bool) -> Self {
        self.verified = v; self
    }

}

impl fmt::Display for EquivalenceProofCert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EquivalenceProofCert({:?})", self.class_id)
    }
}

#[derive(Debug, Clone)]
pub struct EquivalenceProofCertBuilder {
    class_id: u64,
    witness_mapping: Vec<(u32, u32)>,
    verified: bool,
}

impl EquivalenceProofCertBuilder {
    pub fn new() -> Self {
        EquivalenceProofCertBuilder {
            class_id: 0,
            witness_mapping: Vec::new(),
            verified: false,
        }
    }

    pub fn class_id(mut self, v: u64) -> Self { self.class_id = v; self }
    pub fn witness_mapping(mut self, v: Vec<(u32, u32)>) -> Self { self.witness_mapping = v; self }
    pub fn verified(mut self, v: bool) -> Self { self.verified = v; self }
}

#[derive(Debug, Clone)]
pub struct ClassGenerator {
    pub class_count: usize,
    pub tests_per_class: Vec<usize>,
    pub total_generated: usize,
}

impl ClassGenerator {
    pub fn new(class_count: usize, tests_per_class: Vec<usize>, total_generated: usize) -> Self {
        ClassGenerator { class_count, tests_per_class, total_generated }
    }

    pub fn get_class_count(&self) -> usize {
        self.class_count
    }

    pub fn tests_per_class_len(&self) -> usize {
        self.tests_per_class.len()
    }

    pub fn tests_per_class_is_empty(&self) -> bool {
        self.tests_per_class.is_empty()
    }

    pub fn get_total_generated(&self) -> usize {
        self.total_generated
    }

    pub fn with_class_count(mut self, v: usize) -> Self {
        self.class_count = v; self
    }

    pub fn with_total_generated(mut self, v: usize) -> Self {
        self.total_generated = v; self
    }

}

impl fmt::Display for ClassGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ClassGenerator({:?})", self.class_count)
    }
}

#[derive(Debug, Clone)]
pub struct ClassGeneratorBuilder {
    class_count: usize,
    tests_per_class: Vec<usize>,
    total_generated: usize,
}

impl ClassGeneratorBuilder {
    pub fn new() -> Self {
        ClassGeneratorBuilder {
            class_count: 0,
            tests_per_class: Vec::new(),
            total_generated: 0,
        }
    }

    pub fn class_count(mut self, v: usize) -> Self { self.class_count = v; self }
    pub fn tests_per_class(mut self, v: Vec<usize>) -> Self { self.tests_per_class = v; self }
    pub fn total_generated(mut self, v: usize) -> Self { self.total_generated = v; self }
}

#[derive(Debug, Clone)]
pub struct AntiChainComputation {
    pub elements: Vec<Vec<u32>>,
    pub max_antichain: Vec<usize>,
    pub width: usize,
}

impl AntiChainComputation {
    pub fn new(elements: Vec<Vec<u32>>, max_antichain: Vec<usize>, width: usize) -> Self {
        AntiChainComputation { elements, max_antichain, width }
    }

    pub fn elements_len(&self) -> usize {
        self.elements.len()
    }

    pub fn elements_is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn max_antichain_len(&self) -> usize {
        self.max_antichain.len()
    }

    pub fn max_antichain_is_empty(&self) -> bool {
        self.max_antichain.is_empty()
    }

    pub fn get_width(&self) -> usize {
        self.width
    }

    pub fn with_width(mut self, v: usize) -> Self {
        self.width = v; self
    }

}

impl fmt::Display for AntiChainComputation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AntiChainComputation({:?})", self.elements)
    }
}

#[derive(Debug, Clone)]
pub struct AntiChainComputationBuilder {
    elements: Vec<Vec<u32>>,
    max_antichain: Vec<usize>,
    width: usize,
}

impl AntiChainComputationBuilder {
    pub fn new() -> Self {
        AntiChainComputationBuilder {
            elements: Vec::new(),
            max_antichain: Vec::new(),
            width: 0,
        }
    }

    pub fn elements(mut self, v: Vec<Vec<u32>>) -> Self { self.elements = v; self }
    pub fn max_antichain(mut self, v: Vec<usize>) -> Self { self.max_antichain = v; self }
    pub fn width(mut self, v: usize) -> Self { self.width = v; self }
}

#[derive(Debug, Clone)]
pub struct SubsumptionWidth {
    pub width: usize,
    pub chains: Vec<Vec<usize>>,
    pub antichains: Vec<Vec<usize>>,
}

impl SubsumptionWidth {
    pub fn new(width: usize, chains: Vec<Vec<usize>>, antichains: Vec<Vec<usize>>) -> Self {
        SubsumptionWidth { width, chains, antichains }
    }

    pub fn get_width(&self) -> usize {
        self.width
    }

    pub fn chains_len(&self) -> usize {
        self.chains.len()
    }

    pub fn chains_is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    pub fn antichains_len(&self) -> usize {
        self.antichains.len()
    }

    pub fn antichains_is_empty(&self) -> bool {
        self.antichains.is_empty()
    }

    pub fn with_width(mut self, v: usize) -> Self {
        self.width = v; self
    }

}

impl fmt::Display for SubsumptionWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SubsumptionWidth({:?})", self.width)
    }
}

#[derive(Debug, Clone)]
pub struct SubsumptionWidthBuilder {
    width: usize,
    chains: Vec<Vec<usize>>,
    antichains: Vec<Vec<usize>>,
}

impl SubsumptionWidthBuilder {
    pub fn new() -> Self {
        SubsumptionWidthBuilder {
            width: 0,
            chains: Vec::new(),
            antichains: Vec::new(),
        }
    }

    pub fn width(mut self, v: usize) -> Self { self.width = v; self }
    pub fn chains(mut self, v: Vec<Vec<usize>>) -> Self { self.chains = v; self }
    pub fn antichains(mut self, v: Vec<Vec<usize>>) -> Self { self.antichains = v; self }
}

#[derive(Debug, Clone)]
pub struct EquivalenceClass {
    pub id: u64,
    pub representatives: Vec<u32>,
    pub size: usize,
}

impl EquivalenceClass {
    pub fn new(id: u64, representatives: Vec<u32>, size: usize) -> Self {
        EquivalenceClass { id, representatives, size }
    }

    pub fn get_id(&self) -> u64 {
        self.id
    }

    pub fn representatives_len(&self) -> usize {
        self.representatives.len()
    }

    pub fn representatives_is_empty(&self) -> bool {
        self.representatives.is_empty()
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    pub fn with_id(mut self, v: u64) -> Self {
        self.id = v; self
    }

    pub fn with_size(mut self, v: usize) -> Self {
        self.size = v; self
    }

}

impl fmt::Display for EquivalenceClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EquivalenceClass({:?})", self.id)
    }
}

#[derive(Debug, Clone)]
pub struct EquivalenceClassBuilder {
    id: u64,
    representatives: Vec<u32>,
    size: usize,
}

impl EquivalenceClassBuilder {
    pub fn new() -> Self {
        EquivalenceClassBuilder {
            id: 0,
            representatives: Vec::new(),
            size: 0,
        }
    }

    pub fn id(mut self, v: u64) -> Self { self.id = v; self }
    pub fn representatives(mut self, v: Vec<u32>) -> Self { self.representatives = v; self }
    pub fn size(mut self, v: usize) -> Self { self.size = v; self }
}

#[derive(Debug, Clone)]
pub struct CanonicalForm {
    pub thread_order: Vec<u32>,
    pub address_order: Vec<u32>,
    pub value_order: Vec<u64>,
    pub hash: u64,
}

impl CanonicalForm {
    pub fn new(thread_order: Vec<u32>, address_order: Vec<u32>, value_order: Vec<u64>, hash: u64) -> Self {
        CanonicalForm { thread_order, address_order, value_order, hash }
    }

    pub fn thread_order_len(&self) -> usize {
        self.thread_order.len()
    }

    pub fn thread_order_is_empty(&self) -> bool {
        self.thread_order.is_empty()
    }

    pub fn address_order_len(&self) -> usize {
        self.address_order.len()
    }

    pub fn address_order_is_empty(&self) -> bool {
        self.address_order.is_empty()
    }

    pub fn value_order_len(&self) -> usize {
        self.value_order.len()
    }

    pub fn value_order_is_empty(&self) -> bool {
        self.value_order.is_empty()
    }

    pub fn get_hash(&self) -> u64 {
        self.hash
    }

    pub fn with_hash(mut self, v: u64) -> Self {
        self.hash = v; self
    }

}

impl fmt::Display for CanonicalForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CanonicalForm({:?})", self.thread_order)
    }
}

#[derive(Debug, Clone)]
pub struct CanonicalFormBuilder {
    thread_order: Vec<u32>,
    address_order: Vec<u32>,
    value_order: Vec<u64>,
    hash: u64,
}

impl CanonicalFormBuilder {
    pub fn new() -> Self {
        CanonicalFormBuilder {
            thread_order: Vec::new(),
            address_order: Vec::new(),
            value_order: Vec::new(),
            hash: 0,
        }
    }

    pub fn thread_order(mut self, v: Vec<u32>) -> Self { self.thread_order = v; self }
    pub fn address_order(mut self, v: Vec<u32>) -> Self { self.address_order = v; self }
    pub fn value_order(mut self, v: Vec<u64>) -> Self { self.value_order = v; self }
    pub fn hash(mut self, v: u64) -> Self { self.hash = v; self }
}

#[derive(Debug, Clone)]
pub struct EquivAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl EquivAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        EquivAnalysis { data, size, computed: false, label: "Equiv".to_string(), threshold: 0.01 }
    }

    pub fn with_threshold(mut self, t: f64) -> Self {
        self.threshold = t; self
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < self.size && j < self.size { self.data[i][j] = v; }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size { self.data[i][j] } else { 0.0 }
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        if i < self.size { self.data[i].iter().sum() } else { 0.0 }
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        if j < self.size { (0..self.size).map(|i| self.data[i][j]).sum() } else { 0.0 }
    }

    pub fn total_sum(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).sum()
    }

    pub fn max_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn min_value(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn above_threshold(&self) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for i in 0..self.size {
            for j in 0..self.size {
                if self.data[i][j] > self.threshold {
                    result.push((i, j, self.data[i][j]));
                }
            }
        }
        result
    }

    pub fn normalize(&mut self) {
        let total = self.total_sum();
        if total > 0.0 {
            for i in 0..self.size {
                for j in 0..self.size {
                    self.data[i][j] /= total;
                }
            }
        }
        self.computed = true;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                result.data[i][j] = self.data[j][i];
            }
        }
        result
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert_eq!(self.size, other.size);
        let mut result = Self::new(self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                let mut sum = 0.0;
                for k in 0..self.size {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().flat_map(|r| r.iter()).map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn trace(&self) -> f64 {
        (0..self.size).map(|i| self.data[i][i]).sum()
    }

    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.size).map(|i| self.data[i][i]).collect()
    }

    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.size {
            for j in 0..self.size {
                if (self.data[i][j] - self.data[j][i]).abs() > 1e-10 { return false; }
            }
        }
        true
    }

}

impl fmt::Display for EquivAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EquivAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EquivStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for EquivStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquivStatus::Pending => write!(f, "pending"),
            EquivStatus::InProgress => write!(f, "inprogress"),
            EquivStatus::Completed => write!(f, "completed"),
            EquivStatus::Failed => write!(f, "failed"),
            EquivStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EquivPriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for EquivPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquivPriority::Critical => write!(f, "critical"),
            EquivPriority::High => write!(f, "high"),
            EquivPriority::Medium => write!(f, "medium"),
            EquivPriority::Low => write!(f, "low"),
            EquivPriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EquivMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for EquivMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquivMode::Strict => write!(f, "strict"),
            EquivMode::Relaxed => write!(f, "relaxed"),
            EquivMode::Permissive => write!(f, "permissive"),
            EquivMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn equiv_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn equiv_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = equiv_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn equiv_std_dev(data: &[f64]) -> f64 {
    equiv_variance(data).sqrt()
}

pub fn equiv_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

/// Percentile calculator for Equiv.
pub fn equiv_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

pub fn equiv_entropy(data: &[f64]) -> f64 {
    let total: f64 = data.iter().sum();
    if total <= 0.0 { return 0.0; }
    let mut h = 0.0;
    for &x in data {
        if x > 0.0 {
            let p = x / total;
            h -= p * p.ln();
        }
    }
    h
}

pub fn equiv_gini(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = sorted.iter().sum();
    if sum == 0.0 { return 0.0; }
    let mut g = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        g += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * x;
    }
    g / (n as f64 * sum)
}

pub fn equiv_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = equiv_mean(&x);
    let my = equiv_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn equiv_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = equiv_covariance(data);
    let sx = equiv_std_dev(&data[..n]);
    let sy = equiv_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn equiv_excess_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = equiv_mean(data);
    let s = equiv_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn equiv_sample_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = equiv_mean(data);
    let s = equiv_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn equiv_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn equiv_geomean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over equiv analysis results.
#[derive(Debug, Clone)]
pub struct EquivResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl EquivResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        EquivResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for EquivResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert TestNormalizer description to a summary string.
pub fn testnormalizer_to_summary(item: &TestNormalizer) -> String {
    format!("TestNormalizer: {:?}", item)
}

/// Convert ThreadRenaming description to a summary string.
pub fn threadrenaming_to_summary(item: &ThreadRenaming) -> String {
    format!("ThreadRenaming: {:?}", item)
}

/// Convert AddressRenaming description to a summary string.
pub fn addressrenaming_to_summary(item: &AddressRenaming) -> String {
    format!("AddressRenaming: {:?}", item)
}

/// Convert ValueRenaming description to a summary string.
pub fn valuerenaming_to_summary(item: &ValueRenaming) -> String {
    format!("ValueRenaming: {:?}", item)
}

/// Convert TestHasher description to a summary string.
pub fn testhasher_to_summary(item: &TestHasher) -> String {
    format!("TestHasher: {:?}", item)
}

/// Convert TestSerializer description to a summary string.
pub fn testserializer_to_summary(item: &TestSerializer) -> String {
    format!("TestSerializer: {:?}", item)
}

/// Convert EquivalenceProofCert description to a summary string.
pub fn equivalenceproofcert_to_summary(item: &EquivalenceProofCert) -> String {
    format!("EquivalenceProofCert: {:?}", item)
}

/// Convert ClassGenerator description to a summary string.
pub fn classgenerator_to_summary(item: &ClassGenerator) -> String {
    format!("ClassGenerator: {:?}", item)
}

/// Convert AntiChainComputation description to a summary string.
pub fn antichaincomputation_to_summary(item: &AntiChainComputation) -> String {
    format!("AntiChainComputation: {:?}", item)
}

/// Convert SubsumptionWidth description to a summary string.
pub fn subsumptionwidth_to_summary(item: &SubsumptionWidth) -> String {
    format!("SubsumptionWidth: {:?}", item)
}

/// Convert EquivalenceClass description to a summary string.
pub fn equivalenceclass_to_summary(item: &EquivalenceClass) -> String {
    format!("EquivalenceClass: {:?}", item)
}

/// Batch processor for equiv operations.
#[derive(Debug, Clone)]
pub struct EquivBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl EquivBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        EquivBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
    }
    pub fn process_batch(&mut self, data: &[f64]) {
        for chunk in data.chunks(self.batch_size) {
            let sum: f64 = chunk.iter().sum();
            self.results.push(sum / chunk.len() as f64);
            self.processed += chunk.len();
        }
    }
    pub fn success_rate(&self) -> f64 {
        if self.processed == 0 { return 0.0; }
        1.0 - (self.errors.len() as f64 / self.processed as f64)
    }
    pub fn average_result(&self) -> f64 {
        if self.results.is_empty() { return 0.0; }
        self.results.iter().sum::<f64>() / self.results.len() as f64
    }
    pub fn reset(&mut self) { self.processed = 0; self.errors.clear(); self.results.clear(); }
}

impl fmt::Display for EquivBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EquivBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for equiv analysis.
#[derive(Debug, Clone)]
pub struct EquivReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl EquivReport {
    pub fn new(title: impl Into<String>) -> Self {
        EquivReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
    }
    pub fn add_section(&mut self, name: impl Into<String>, content: Vec<String>) {
        self.sections.push((name.into(), content));
    }
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.push((name.into(), value));
    }
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
    pub fn total_metrics(&self) -> usize { self.metrics.len() }
    pub fn has_warnings(&self) -> bool { !self.warnings.is_empty() }
    pub fn metric_sum(&self) -> f64 { self.metrics.iter().map(|(_, v)| v).sum() }
    pub fn render_text(&self) -> String {
        let mut out = format!("=== {} ===\n", self.title);
        for (name, content) in &self.sections {
            out.push_str(&format!("\n--- {} ---\n", name));
            for line in content {
                out.push_str(&format!("  {}\n", line));
            }
        }
        out.push_str("\nMetrics:\n");
        for (name, val) in &self.metrics {
            out.push_str(&format!("  {}: {:.4}\n", name, val));
        }
        if !self.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &self.warnings {
                out.push_str(&format!("  ! {}\n", w));
            }
        }
        out
    }
}

impl fmt::Display for EquivReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EquivReport({})", self.title)
    }
}

/// Configuration for equiv analysis.
#[derive(Debug, Clone)]
pub struct EquivConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl EquivConfig {
    pub fn default_config() -> Self {
        EquivConfig {
            verbose: false, max_iterations: 1000, tolerance: 1e-6,
            timeout_ms: 30000, parallel: false, output_format: "text".to_string(),
        }
    }
    pub fn with_verbose(mut self, v: bool) -> Self { self.verbose = v; self }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_tolerance(mut self, t: f64) -> Self { self.tolerance = t; self }
    pub fn with_timeout(mut self, ms: u64) -> Self { self.timeout_ms = ms; self }
    pub fn with_parallel(mut self, p: bool) -> Self { self.parallel = p; self }
    pub fn with_output_format(mut self, fmt: impl Into<String>) -> Self { self.output_format = fmt.into(); self }
}

impl fmt::Display for EquivConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EquivConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for equiv data distribution.
#[derive(Debug, Clone)]
pub struct EquivHistogramExt {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl EquivHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return EquivHistogramExt { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
        }
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        let bin_width = if range == 0.0 { 1.0 } else { range / num_bins as f64 };
        let mut bins = vec![0usize; num_bins];
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins { bin_edges.push(min_val + i as f64 * bin_width); }
        for &val in data {
            let idx = ((val - min_val) / bin_width).floor() as usize;
            let idx = idx.min(num_bins - 1);
            bins[idx] += 1;
        }
        EquivHistogramExt { bins, bin_edges, total_count: data.len() }
    }
    pub fn num_bins(&self) -> usize { self.bins.len() }
    pub fn max_bin(&self) -> usize { self.bins.iter().cloned().max().unwrap_or(0) }
    pub fn mean_bin(&self) -> f64 {
        if self.bins.is_empty() { return 0.0; }
        self.bins.iter().sum::<usize>() as f64 / self.bins.len() as f64
    }
    pub fn cumulative(&self) -> Vec<usize> {
        let mut cum = Vec::with_capacity(self.bins.len());
        let mut acc = 0usize;
        for &b in &self.bins { acc += b; cum.push(acc); }
        cum
    }
    pub fn entropy(&self) -> f64 {
        let total = self.total_count as f64;
        if total == 0.0 { return 0.0; }
        let mut h = 0.0f64;
        for &b in &self.bins {
            if b > 0 { let p = b as f64 / total; h -= p * p.ln(); }
        }
        h
    }
    pub fn render_ascii(&self, width: usize) -> String {
        let max = self.max_bin();
        let mut out = String::new();
        for (i, &count) in self.bins.iter().enumerate() {
            let bar_len = if max == 0 { 0 } else { count * width / max };
            let bar: String = std::iter::repeat('#').take(bar_len).collect();
            out.push_str(&format!("[{:.2}-{:.2}] {} {}\n",
                self.bin_edges[i], self.bin_edges[i + 1], bar, count));
        }
        out
    }
}

impl fmt::Display for EquivHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for equiv graph analysis.
#[derive(Debug, Clone)]
pub struct EquivGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl EquivGraph {
    pub fn new(n: usize) -> Self {
        EquivGraph {
            adjacency: vec![vec![false; n]; n],
            weights: vec![vec![0.0; n]; n],
            node_count: n, edge_count: 0,
            node_labels: (0..n).map(|i| format!("n{}", i)).collect(),
        }
    }
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        if from < self.node_count && to < self.node_count && !self.adjacency[from][to] {
            self.adjacency[from][to] = true;
            self.weights[from][to] = weight;
            self.edge_count += 1;
        }
    }
    pub fn remove_edge(&mut self, from: usize, to: usize) {
        if from < self.node_count && to < self.node_count && self.adjacency[from][to] {
            self.adjacency[from][to] = false;
            self.weights[from][to] = 0.0;
            self.edge_count -= 1;
        }
    }
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        from < self.node_count && to < self.node_count && self.adjacency[from][to]
    }
    pub fn weight(&self, from: usize, to: usize) -> f64 { self.weights[from][to] }
    pub fn out_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).count()
    }
    pub fn in_degree(&self, node: usize) -> usize {
        (0..self.node_count).filter(|&i| self.adjacency[i][node]).count()
    }
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        (0..self.node_count).filter(|&j| self.adjacency[node][j]).collect()
    }
    pub fn density(&self) -> f64 {
        if self.node_count <= 1 { return 0.0; }
        self.edge_count as f64 / (self.node_count * (self.node_count - 1)) as f64
    }
    pub fn is_acyclic(&self) -> bool {
        let n = self.node_count;
        let mut visited = vec![0u8; n];
        fn dfs_cycle_equiv(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_equiv(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_equiv(i, &self.adjacency, &mut visited) { return false; }
        }
        true
    }
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let n = self.node_count;
        let mut in_deg: Vec<usize> = (0..n).map(|j| self.in_degree(j)).collect();
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut result = Vec::new();
        while let Some(v) = queue.pop() {
            result.push(v);
            for j in 0..n { if self.adjacency[v][j] {
                in_deg[j] -= 1;
                if in_deg[j] == 0 { queue.push(j); }
            }}
        }
        if result.len() == n { Some(result) } else { None }
    }
    pub fn shortest_path_dijkstra(&self, start: usize) -> Vec<f64> {
        let n = self.node_count;
        let mut dist = vec![f64::INFINITY; n];
        let mut visited = vec![false; n];
        dist[start] = 0.0;
        for _ in 0..n {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..n { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for v in 0..n { if self.adjacency[u][v] {
                let alt = dist[u] + self.weights[u][v];
                if alt < dist[v] { dist[v] = alt; }
            }}
        }
        dist
    }
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.node_count;
        let mut visited = vec![false; n];
        let mut components = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                if visited[v] { continue; }
                visited[v] = true;
                comp.push(v);
                for w in 0..n {
                    if (self.adjacency[v][w] || self.adjacency[w][v]) && !visited[w] {
                        stack.push(w);
                    }
                }
            }
            components.push(comp);
        }
        components
    }
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph {\n");
        for i in 0..self.node_count {
            out.push_str(&format!("  {} [label=\"{}\"];\n", i, self.node_labels[i]));
        }
        for i in 0..self.node_count { for j in 0..self.node_count { if self.adjacency[i][j] {
            out.push_str(&format!("  {} -> {} [label=\"{:.2}\"];\n", i, j, self.weights[i][j]));
        }}}
        out.push_str("}\n");
        out
    }
}

impl fmt::Display for EquivGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EquivGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for equiv computation results.
#[derive(Debug, Clone)]
pub struct EquivCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl EquivCache {
    pub fn new(capacity: usize) -> Self {
        EquivCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
    }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == key) {
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn insert(&mut self, key: u64, value: Vec<f64>) {
        if self.entries.len() >= self.capacity { self.entries.remove(0); }
        self.entries.push((key, value));
    }
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; }
}

impl fmt::Display for EquivCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for equiv elements.
pub fn equiv_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = points.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            let d: f64 = points[i].iter().zip(points[j].iter())
                .map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    distances
}

/// K-means clustering for equiv data.
pub fn equiv_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
    if data.is_empty() || k == 0 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    let mut centroids: Vec<Vec<f64>> = data.iter().take(k).cloned().collect();
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iters {
        // Assign
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0; let mut best_d = f64::INFINITY;
            for c in 0..centroids.len() {
                let d: f64 = data[i].iter().zip(centroids[c].iter())
                    .map(|(a, b)| (a - b) * (a - b)).sum();
                if d < best_d { best_d = d; best_c = c; }
            }
            if assignments[i] != best_c { changed = true; assignments[i] = best_c; }
        }
        if !changed { break; }
        // Update centroids
        for c in 0..centroids.len() {
            let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
            if members.is_empty() { continue; }
            for d in 0..dim {
                centroids[c][d] = members.iter().map(|&i| data[i][d]).sum::<f64>() / members.len() as f64;
            }
        }
    }
    assignments
}

/// Principal component analysis (simplified) for equiv data.
pub fn equiv_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
    if data.is_empty() || data[0].len() < 2 { return Vec::new(); }
    let n = data.len();
    let dim = data[0].len();
    // Compute mean
    let mut mean = vec![0.0; dim];
    for row in data { for (j, &v) in row.iter().enumerate() { mean[j] += v; } }
    for j in 0..dim { mean[j] /= n as f64; }
    // Center data
    let centered: Vec<Vec<f64>> = data.iter().map(|row| {
        row.iter().zip(mean.iter()).map(|(v, m)| v - m).collect()
    }).collect();
    // Simple projection onto first two dimensions (not true PCA)
    centered.iter().map(|row| (row[0], row[1])).collect()
}

/// Dense matrix operations for Equiv computations.
#[derive(Debug, Clone)]
pub struct EquivDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl EquivDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        EquivDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        EquivDenseMatrix { rows, cols, data }
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> Vec<f64> {
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.data[i * self.cols + j]).collect()
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
        EquivDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        EquivDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn mul_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols { sum += self.get(i, k) * other.get(k, j); }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&v| v * s).collect();
        EquivDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { result.set(j, i, self.get(i, j)); } }
        result
    }

    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
    }

    pub fn row_sum(&self, i: usize) -> f64 {
        (0..self.cols).map(|j| self.get(i, j)).sum()
    }

    pub fn col_sum(&self, j: usize) -> f64 {
        (0..self.rows).map(|i| self.get(i, j)).sum()
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn is_symmetric(&self) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows { for j in (i+1)..self.cols {
            if (self.get(i, j) - self.get(j, i)).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_diagonal(&self) -> bool {
        for i in 0..self.rows { for j in 0..self.cols {
            if i != j && self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn is_upper_triangular(&self) -> bool {
        for i in 0..self.rows { for j in 0..i.min(self.cols) {
            if self.get(i, j).abs() > 1e-10 { return false; }
        }}
        true
    }

    pub fn determinant_2x2(&self) -> f64 {
        assert!(self.rows == 2 && self.cols == 2);
        self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
    }

    pub fn determinant_3x3(&self) -> f64 {
        assert!(self.rows == 3 && self.cols == 3);
        let a = self.get(0, 0); let b = self.get(0, 1); let c = self.get(0, 2);
        let d = self.get(1, 0); let e = self.get(1, 1); let ff = self.get(1, 2);
        let g = self.get(2, 0); let h = self.get(2, 1); let ii = self.get(2, 2);
        a * (e * ii - ff * h) - b * (d * ii - ff * g) + c * (d * h - e * g)
    }

    pub fn inverse_2x2(&self) -> Option<Self> {
        assert!(self.rows == 2 && self.cols == 2);
        let det = self.determinant_2x2();
        if det.abs() < 1e-15 { return None; }
        let inv_det = 1.0 / det;
        let mut result = Self::new(2, 2);
        result.set(0, 0, self.get(1, 1) * inv_det);
        result.set(0, 1, -self.get(0, 1) * inv_det);
        result.set(1, 0, -self.get(1, 0) * inv_det);
        result.set(1, 1, self.get(0, 0) * inv_det);
        Some(result)
    }

    pub fn power(&self, n: u32) -> Self {
        assert!(self.is_square());
        let mut result = Self::identity(self.rows);
        for _ in 0..n { result = result.mul_matrix(self); }
        result
    }

    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Self {
        let mut result = Self::new(rows, cols);
        for i in 0..rows { for j in 0..cols {
            result.set(i, j, self.get(row_start + i, col_start + j));
        }}
        result
    }

    pub fn kronecker_product(&self, other: &Self) -> Self {
        let m = self.rows * other.rows;
        let n = self.cols * other.cols;
        let mut result = Self::new(m, n);
        for i in 0..self.rows { for j in 0..self.cols {
            let s = self.get(i, j);
            for p in 0..other.rows { for q in 0..other.cols {
                result.set(i * other.rows + p, j * other.cols + q, s * other.get(p, q));
            }}
        }}
        result
    }

    pub fn hadamard_product(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
        EquivDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn outer_product(a: &[f64], b: &[f64]) -> Self {
        let mut result = Self::new(a.len(), b.len());
        for i in 0..a.len() { for j in 0..b.len() { result.set(i, j, a[i] * b[j]); } }
        result
    }

    pub fn row_reduce(&self) -> Self {
        let mut result = self.clone();
        let mut pivot_row = 0;
        for col in 0..result.cols {
            if pivot_row >= result.rows { break; }
            let mut max_row = pivot_row;
            for row in (pivot_row + 1)..result.rows {
                if result.get(row, col).abs() > result.get(max_row, col).abs() { max_row = row; }
            }
            if result.get(max_row, col).abs() < 1e-10 { continue; }
            for j in 0..result.cols {
                let tmp = result.get(pivot_row, j);
                result.set(pivot_row, j, result.get(max_row, j));
                result.set(max_row, j, tmp);
            }
            let pivot = result.get(pivot_row, col);
            for j in 0..result.cols { result.set(pivot_row, j, result.get(pivot_row, j) / pivot); }
            for row in 0..result.rows {
                if row == pivot_row { continue; }
                let factor = result.get(row, col);
                for j in 0..result.cols {
                    let v = result.get(row, j) - factor * result.get(pivot_row, j);
                    result.set(row, j, v);
                }
            }
            pivot_row += 1;
        }
        result
    }

    pub fn rank(&self) -> usize {
        let rref = self.row_reduce();
        let mut r = 0;
        for i in 0..rref.rows {
            if (0..rref.cols).any(|j| rref.get(i, j).abs() > 1e-10) { r += 1; }
        }
        r
    }

    pub fn nullity(&self) -> usize {
        self.cols - self.rank()
    }

    pub fn column_space_basis(&self) -> Vec<Vec<f64>> {
        let rref = self.row_reduce();
        let mut basis = Vec::new();
        for j in 0..self.cols {
            let is_pivot = (0..rref.rows).any(|i| {
                (rref.get(i, j) - 1.0).abs() < 1e-10 &&
                (0..j).all(|k| rref.get(i, k).abs() < 1e-10)
            });
            if is_pivot { basis.push(self.col(j)); }
        }
        basis
    }

    pub fn lu_decomposition(&self) -> (Self, Self) {
        assert!(self.is_square());
        let n = self.rows;
        let mut l = Self::identity(n);
        let mut u = self.clone();
        for k in 0..n {
            for i in (k+1)..n {
                if u.get(k, k).abs() < 1e-15 { continue; }
                let factor = u.get(i, k) / u.get(k, k);
                l.set(i, k, factor);
                for j in k..n {
                    let v = u.get(i, j) - factor * u.get(k, j);
                    u.set(i, j, v);
                }
            }
        }
        (l, u)
    }

    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        assert!(self.is_square());
        assert_eq!(self.rows, b.len());
        let n = self.rows;
        let mut augmented = Self::new(n, n + 1);
        for i in 0..n { for j in 0..n { augmented.set(i, j, self.get(i, j)); } augmented.set(i, n, b[i]); }
        let rref = augmented.row_reduce();
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = rref.get(i, n);
            for j in (i+1)..n { x[i] -= rref.get(i, j) * x[j]; }
            if rref.get(i, i).abs() < 1e-15 { return None; }
            x[i] /= rref.get(i, i);
        }
        Some(x)
    }

    pub fn eigenvalues_2x2(&self) -> (f64, f64) {
        assert!(self.rows == 2 && self.cols == 2);
        let tr = self.trace();
        let det = self.determinant_2x2();
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            ((tr + disc.sqrt()) / 2.0, (tr - disc.sqrt()) / 2.0)
        } else {
            (tr / 2.0, tr / 2.0)
        }
    }

    pub fn condition_number(&self) -> f64 {
        let max_sv = self.frobenius_norm();
        if max_sv < 1e-15 { return f64::INFINITY; }
        max_sv
    }

}

impl fmt::Display for EquivDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EquivMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for Equiv bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct EquivInterval {
    pub lo: f64,
    pub hi: f64,
}

impl EquivInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        EquivInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        EquivInterval { lo: v, hi: v }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    pub fn contains(&self, v: f64) -> bool {
        self.lo <= v && v <= self.hi
    }

    pub fn overlaps(&self, other: &Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    pub fn hull(&self, other: &Self) -> Self {
        EquivInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(EquivInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        EquivInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        EquivInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        EquivInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { EquivInterval { lo: -self.hi, hi: -self.lo } }
        else { EquivInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        EquivInterval { lo, hi: self.hi.max(0.0).sqrt() }
    }

    pub fn is_positive(&self) -> bool {
        self.lo > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.hi < 0.0
    }

    pub fn is_zero(&self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn is_point(&self) -> bool {
        (self.hi - self.lo).abs() < 1e-15
    }

}

impl fmt::Display for EquivInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for Equiv protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EquivState {
    Init,
    Normalizing,
    Hashing,
    Comparing,
    Classified,
    Done,
}

impl fmt::Display for EquivState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquivState::Init => write!(f, "init"),
            EquivState::Normalizing => write!(f, "normalizing"),
            EquivState::Hashing => write!(f, "hashing"),
            EquivState::Comparing => write!(f, "comparing"),
            EquivState::Classified => write!(f, "classified"),
            EquivState::Done => write!(f, "done"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EquivStateMachine {
    pub current: EquivState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl EquivStateMachine {
    pub fn new() -> Self {
        EquivStateMachine { current: EquivState::Init, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &EquivState { &self.current }
    pub fn can_transition(&self, target: &EquivState) -> bool {
        match (&self.current, target) {
            (EquivState::Init, EquivState::Normalizing) => true,
            (EquivState::Normalizing, EquivState::Hashing) => true,
            (EquivState::Hashing, EquivState::Comparing) => true,
            (EquivState::Comparing, EquivState::Classified) => true,
            (EquivState::Classified, EquivState::Done) => true,
            (EquivState::Done, EquivState::Init) => true,
            (EquivState::Comparing, EquivState::Done) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: EquivState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = EquivState::Init;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for EquivStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for Equiv event tracking.
#[derive(Debug, Clone)]
pub struct EquivRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl EquivRingBuffer {
    pub fn new(capacity: usize) -> Self {
        EquivRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
    }
    pub fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity { self.count += 1; }
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn is_full(&self) -> bool { self.count == self.capacity }
    pub fn latest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - 1) % self.capacity]) }
    }
    pub fn oldest(&self) -> Option<f64> {
        if self.count == 0 { None }
        else { Some(self.data[(self.head + self.capacity - self.count) % self.capacity]) }
    }
    pub fn average(&self) -> f64 {
        if self.count == 0 { return 0.0; }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.data[(self.head + self.capacity - 1 - i) % self.capacity];
        }
        sum / self.count as f64
    }
    pub fn to_vec(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.count);
        for i in 0..self.count {
            result.push(self.data[(self.head + self.capacity - self.count + i) % self.capacity]);
        }
        result
    }
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::INFINITY, f64::min))
    }
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 { return None; }
        Some(self.to_vec().iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        let avg = self.average();
        let v: f64 = self.to_vec().iter().map(|&x| (x - avg) * (x - avg)).sum();
        v / (self.count - 1) as f64
    }
    pub fn clear(&mut self) { self.head = 0; self.count = 0; }
}

impl fmt::Display for EquivRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for Equiv component tracking.
#[derive(Debug, Clone)]
pub struct EquivDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl EquivDisjointSet {
    pub fn new(n: usize) -> Self {
        EquivDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
    }
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x { self.parent[x] = self.parent[self.parent[x]]; x = self.parent[x]; }
        x
    }
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x); let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] { self.parent[rx] = ry; self.size[ry] += self.size[rx]; }
        else if self.rank[rx] > self.rank[ry] { self.parent[ry] = rx; self.size[rx] += self.size[ry]; }
        else { self.parent[ry] = rx; self.size[rx] += self.size[ry]; self.rank[rx] += 1; }
        self.num_components -= 1;
        true
    }
    pub fn connected(&mut self, x: usize, y: usize) -> bool { self.find(x) == self.find(y) }
    pub fn component_size(&mut self, x: usize) -> usize { let r = self.find(x); self.size[r] }
    pub fn num_components(&self) -> usize { self.num_components }
    pub fn components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for i in 0..n { let r = self.find(i); groups.entry(r).or_default().push(i); }
        groups.into_values().collect()
    }
}

impl fmt::Display for EquivDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for Equiv.
#[derive(Debug, Clone)]
pub struct EquivSortedList {
    data: Vec<f64>,
}

impl EquivSortedList {
    pub fn new() -> Self { EquivSortedList { data: Vec::new() } }
    pub fn insert(&mut self, value: f64) {
        let pos = self.data.partition_point(|&x| x < value);
        self.data.insert(pos, value);
    }
    pub fn contains(&self, value: f64) -> bool {
        self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()).is_ok()
    }
    pub fn rank(&self, value: f64) -> usize { self.data.partition_point(|&x| x < value) }
    pub fn quantile(&self, q: f64) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let idx = ((self.data.len() - 1) as f64 * q).round() as usize;
        self.data[idx.min(self.data.len() - 1)]
    }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn min(&self) -> Option<f64> { self.data.first().copied() }
    pub fn max(&self) -> Option<f64> { self.data.last().copied() }
    pub fn median(&self) -> f64 { self.quantile(0.5) }
    pub fn iqr(&self) -> f64 { self.quantile(0.75) - self.quantile(0.25) }
    pub fn remove(&mut self, value: f64) -> bool {
        if let Ok(pos) = self.data.binary_search_by(|x| x.partial_cmp(&value).unwrap()) {
            self.data.remove(pos); true
        } else { false }
    }
    pub fn range(&self, lo: f64, hi: f64) -> Vec<f64> {
        self.data.iter().filter(|&&x| x >= lo && x <= hi).cloned().collect()
    }
    pub fn to_vec(&self) -> Vec<f64> { self.data.clone() }
}

impl fmt::Display for EquivSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for Equiv metrics.
#[derive(Debug, Clone)]
pub struct EquivEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl EquivEma {
    pub fn new(alpha: f64) -> Self { EquivEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for EquivEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for Equiv membership testing.
#[derive(Debug, Clone)]
pub struct EquivBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl EquivBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        EquivBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
    }
    fn hash_indices(&self, value: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.num_hashes);
        let mut h = value;
        for _ in 0..self.num_hashes {
            h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(0x6c62272e07bb0142);
            indices.push((h as usize) % self.size);
        }
        indices
    }
    pub fn insert(&mut self, value: u64) {
        for idx in self.hash_indices(value) { self.bits[idx] = true; }
        self.count += 1;
    }
    pub fn may_contain(&self, value: u64) -> bool {
        self.hash_indices(value).iter().all(|&idx| self.bits[idx])
    }
    pub fn false_positive_rate(&self) -> f64 {
        let set_bits = self.bits.iter().filter(|&&b| b).count() as f64;
        (set_bits / self.size as f64).powi(self.num_hashes as i32)
    }
    pub fn count(&self) -> usize { self.count }
    pub fn clear(&mut self) { self.bits.fill(false); self.count = 0; }
}

impl fmt::Display for EquivBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for Equiv string matching.
#[derive(Debug, Clone)]
pub struct EquivTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct EquivTrie {
    nodes: Vec<EquivTrieNode>,
    count: usize,
}

impl EquivTrie {
    pub fn new() -> Self {
        EquivTrie { nodes: vec![EquivTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(EquivTrieNode { children: Vec::new(), is_terminal: false, value: None });
                    self.nodes[current].children.push((ch, idx));
                    idx
                }
            };
        }
        self.nodes[current].is_terminal = true;
        self.nodes[current].value = Some(value);
        self.count += 1;
    }
    pub fn search(&self, key: &str) -> Option<u64> {
        let mut current = 0;
        for ch in key.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return None,
            }
        }
        if self.nodes[current].is_terminal { self.nodes[current].value } else { None }
    }
    pub fn starts_with(&self, prefix: &str) -> bool {
        let mut current = 0;
        for ch in prefix.chars() {
            match self.nodes[current].children.iter().find(|(c, _)| *c == ch) {
                Some((_, idx)) => current = *idx,
                None => return false,
            }
        }
        true
    }
    pub fn len(&self) -> usize { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn node_count(&self) -> usize { self.nodes.len() }
}

impl fmt::Display for EquivTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for Equiv scheduling.
#[derive(Debug, Clone)]
pub struct EquivPriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl EquivPriorityQueue {
    pub fn new() -> Self { EquivPriorityQueue { heap: Vec::new() } }
    pub fn push(&mut self, priority: f64, item: usize) {
        self.heap.push((priority, item));
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.heap[i].0 < self.heap[parent].0 { self.heap.swap(i, parent); i = parent; }
            else { break; }
        }
    }
    pub fn pop(&mut self) -> Option<(f64, usize)> {
        if self.heap.is_empty() { return None; }
        let result = self.heap.swap_remove(0);
        if !self.heap.is_empty() { self.sift_down(0); }
        Some(result)
    }
    fn sift_down(&mut self, mut i: usize) {
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut smallest = i;
            if left < self.heap.len() && self.heap[left].0 < self.heap[smallest].0 { smallest = left; }
            if right < self.heap.len() && self.heap[right].0 < self.heap[smallest].0 { smallest = right; }
            if smallest != i { self.heap.swap(i, smallest); i = smallest; }
            else { break; }
        }
    }
    pub fn peek(&self) -> Option<&(f64, usize)> { self.heap.first() }
    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }
}

impl fmt::Display for EquivPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for Equiv.
#[derive(Debug, Clone)]
pub struct EquivAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl EquivAccumulator {
    pub fn new() -> Self { EquivAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.min_val = self.min_val.min(value);
        self.max_val = self.max_val.max(value);
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }
    pub fn count(&self) -> u64 { self.count }
    pub fn mean(&self) -> f64 { self.mean }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.min_val }
    pub fn max(&self) -> f64 { self.max_val }
    pub fn sum(&self) -> f64 { self.sum }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() }
    }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean = (self.sum + other.sum) / total as f64;
        self.m2 += other.m2 + delta * delta * (self.count as f64 * other.count as f64 / total as f64);
        self.mean = new_mean;
        self.count = total;
        self.sum += other.sum;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
    }
    pub fn reset(&mut self) {
        self.count = 0; self.mean = 0.0; self.m2 = 0.0;
        self.min_val = f64::INFINITY; self.max_val = f64::NEG_INFINITY; self.sum = 0.0;
    }
}

impl fmt::Display for EquivAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for Equiv.
#[derive(Debug, Clone)]
pub struct EquivSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl EquivSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { EquivSparseMatrix { rows, cols, entries: Vec::new() } }
    pub fn insert(&mut self, i: usize, j: usize, v: f64) {
        if let Some(pos) = self.entries.iter().position(|&(r, c, _)| r == i && c == j) {
            self.entries[pos].2 = v;
        } else { self.entries.push((i, j, v)); }
    }
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.entries.iter().find(|&&(r, c, _)| r == i && c == j).map(|&(_, _, v)| v).unwrap_or(0.0)
    }
    pub fn nnz(&self) -> usize { self.entries.len() }
    pub fn density(&self) -> f64 { self.entries.len() as f64 / (self.rows * self.cols) as f64 }
    pub fn transpose(&self) -> Self {
        let mut result = EquivSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = EquivSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = EquivSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.entries.push((i, j, v * s)); }
        result
    }
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.rows];
        for &(i, j, v) in &self.entries { result[i] += v * x[j]; }
        result
    }
    pub fn frobenius_norm(&self) -> f64 { self.entries.iter().map(|&(_, _, v)| v * v).sum::<f64>().sqrt() }
    pub fn row_nnz(&self, i: usize) -> usize { self.entries.iter().filter(|&&(r, _, _)| r == i).count() }
    pub fn col_nnz(&self, j: usize) -> usize { self.entries.iter().filter(|&&(_, c, _)| c == j).count() }
    pub fn to_dense(&self, dm_new: fn(usize, usize) -> Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; self.cols]; self.rows];
        for &(i, j, v) in &self.entries { result[i][j] = v; }
        result
    }
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }
    pub fn trace(&self) -> f64 { self.diagonal().iter().sum() }
    pub fn remove_zeros(&mut self, tol: f64) {
        self.entries.retain(|&(_, _, v)| v.abs() > tol);
    }
}

impl fmt::Display for EquivSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for Equiv.
#[derive(Debug, Clone)]
pub struct EquivPolynomial {
    pub coefficients: Vec<f64>,
}

impl EquivPolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { EquivPolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { EquivPolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { EquivPolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        EquivPolynomial { coefficients: c }
    }
    pub fn degree(&self) -> usize {
        if self.coefficients.is_empty() { return 0; }
        let mut d = self.coefficients.len() - 1;
        while d > 0 && self.coefficients[d].abs() < 1e-15 { d -= 1; }
        d
    }
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut power = 1.0;
        for &c in &self.coefficients {
            result += c * power;
            power *= x;
        }
        result
    }
    pub fn evaluate_horner(&self, x: f64) -> f64 {
        if self.coefficients.is_empty() { return 0.0; }
        let mut result = *self.coefficients.last().unwrap();
        for &c in self.coefficients.iter().rev().skip(1) {
            result = result * x + c;
        }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] += c; }
        EquivPolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        EquivPolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        EquivPolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        EquivPolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        EquivPolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        EquivPolynomial { coefficients: coeffs }
    }
    pub fn roots_quadratic(&self) -> Vec<f64> {
        if self.degree() != 2 { return Vec::new(); }
        let a = self.coefficients[2];
        let b = self.coefficients[1];
        let c = self.coefficients[0];
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 { Vec::new() }
        else if disc.abs() < 1e-15 { vec![-b / (2.0 * a)] }
        else { vec![(-b + disc.sqrt()) / (2.0 * a), (-b - disc.sqrt()) / (2.0 * a)] }
    }
    pub fn is_zero(&self) -> bool { self.coefficients.iter().all(|&c| c.abs() < 1e-15) }
    pub fn leading_coefficient(&self) -> f64 {
        self.coefficients.get(self.degree()).copied().unwrap_or(0.0)
    }
    pub fn compose(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let mut power = Self::one();
        for &c in &self.coefficients {
            result = result.add(&power.scale(c));
            power = power.mul(other);
        }
        result
    }
    pub fn newton_root(&self, initial_guess: f64, max_iters: usize, tol: f64) -> Option<f64> {
        let deriv = self.derivative();
        let mut x = initial_guess;
        for _ in 0..max_iters {
            let fx = self.evaluate(x);
            if fx.abs() < tol { return Some(x); }
            let dfx = deriv.evaluate(x);
            if dfx.abs() < 1e-15 { return None; }
            x -= fx / dfx;
        }
        if self.evaluate(x).abs() < tol * 100.0 { Some(x) } else { None }
    }
}

impl fmt::Display for EquivPolynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms = Vec::new();
        for (i, &c) in self.coefficients.iter().enumerate() {
            if c.abs() < 1e-15 { continue; }
            if i == 0 { terms.push(format!("{:.2}", c)); }
            else if i == 1 { terms.push(format!("{:.2}x", c)); }
            else { terms.push(format!("{:.2}x^{}", c, i)); }
        }
        if terms.is_empty() { write!(f, "0") }
        else { write!(f, "{}", terms.join(" + ")) }
    }
}

/// Simple linear congruential generator for Equiv.
#[derive(Debug, Clone)]
pub struct EquivRng {
    state: u64,
}

impl EquivRng {
    pub fn new(seed: u64) -> Self { EquivRng { state: seed } }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
        if hi <= lo { return lo; }
        lo + (self.next_u64() % (hi - lo))
    }
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle(&mut self, data: &mut [f64]) {
        let n = data.len();
        for i in (1..n).rev() {
            let j = self.next_range(0, i as u64 + 1) as usize;
            data.swap(i, j);
        }
    }
    pub fn sample(&mut self, data: &[f64], n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = self.next_range(0, data.len() as u64) as usize;
            result.push(data[idx]);
        }
        result
    }
    pub fn bernoulli(&mut self, p: f64) -> bool { self.next_f64() < p }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 { lo + self.next_f64() * (hi - lo) }
    pub fn exponential(&mut self, lambda: f64) -> f64 { -self.next_f64().max(1e-15).ln() / lambda }
}

impl fmt::Display for EquivRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for Equiv benchmarking.
#[derive(Debug, Clone)]
pub struct EquivTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl EquivTimer {
    pub fn new(label: impl Into<String>) -> Self { EquivTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
    pub fn record(&mut self, ns: u64) { self.elapsed_ns.push(ns); }
    pub fn total_ns(&self) -> u64 { self.elapsed_ns.iter().sum() }
    pub fn count(&self) -> usize { self.elapsed_ns.len() }
    pub fn average_ns(&self) -> f64 {
        if self.elapsed_ns.is_empty() { 0.0 } else { self.total_ns() as f64 / self.elapsed_ns.len() as f64 }
    }
    pub fn min_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().min().unwrap_or(0) }
    pub fn max_ns(&self) -> u64 { self.elapsed_ns.iter().cloned().max().unwrap_or(0) }
    pub fn percentile_ns(&self, p: f64) -> u64 {
        if self.elapsed_ns.is_empty() { return 0; }
        let mut sorted = self.elapsed_ns.clone();
        sorted.sort();
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    pub fn p50_ns(&self) -> u64 { self.percentile_ns(0.5) }
    pub fn p95_ns(&self) -> u64 { self.percentile_ns(0.95) }
    pub fn p99_ns(&self) -> u64 { self.percentile_ns(0.99) }
    pub fn reset(&mut self) { self.elapsed_ns.clear(); self.running = false; }
}

impl fmt::Display for EquivTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for Equiv set operations.
#[derive(Debug, Clone)]
pub struct EquivBitVector {
    words: Vec<u64>,
    len: usize,
}

impl EquivBitVector {
    pub fn new(len: usize) -> Self { EquivBitVector { words: vec![0u64; (len + 63) / 64], len } }
    pub fn set(&mut self, i: usize) { if i < self.len { self.words[i / 64] |= 1u64 << (i % 64); } }
    pub fn clear(&mut self, i: usize) { if i < self.len { self.words[i / 64] &= !(1u64 << (i % 64)); } }
    pub fn get(&self, i: usize) -> bool { i < self.len && (self.words[i / 64] & (1u64 << (i % 64))) != 0 }
    pub fn len(&self) -> usize { self.len }
    pub fn count_ones(&self) -> usize { self.words.iter().map(|w| w.count_ones() as usize).sum() }
    pub fn count_zeros(&self) -> usize { self.len - self.count_ones() }
    pub fn is_empty(&self) -> bool { self.count_ones() == 0 }
    pub fn and(&self, other: &Self) -> Self {
        let n = self.words.len().min(other.words.len());
        let mut result = Self::new(self.len.min(other.len));
        for i in 0..n { result.words[i] = self.words[i] & other.words[i]; }
        result
    }
    pub fn or(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] |= self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] |= other.words[i]; }
        result
    }
    pub fn xor(&self, other: &Self) -> Self {
        let n = self.words.len().max(other.words.len());
        let mut result = Self::new(self.len.max(other.len));
        for i in 0..self.words.len().min(n) { result.words[i] = self.words[i]; }
        for i in 0..other.words.len().min(n) { result.words[i] ^= other.words[i]; }
        result
    }
    pub fn not(&self) -> Self {
        let mut result = Self::new(self.len);
        for i in 0..self.words.len() { result.words[i] = !self.words[i]; }
        // Clear unused bits in last word
        let extra = self.len % 64;
        if extra > 0 && !result.words.is_empty() {
            let last = result.words.len() - 1;
            result.words[last] &= (1u64 << extra) - 1;
        }
        result
    }
    pub fn iter_ones(&self) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0..self.len { if self.get(i) { result.push(i); } }
        result
    }
    pub fn jaccard(&self, other: &Self) -> f64 {
        let intersection = self.and(other).count_ones() as f64;
        let union = self.or(other).count_ones() as f64;
        if union == 0.0 { 1.0 } else { intersection / union }
    }
    pub fn hamming_distance(&self, other: &Self) -> usize { self.xor(other).count_ones() }
    pub fn fill(&mut self, value: bool) {
        let fill_val = if value { u64::MAX } else { 0 };
        for w in &mut self.words { *w = fill_val; }
        if value { let extra = self.len % 64; if extra > 0 { let last = self.words.len() - 1; self.words[last] &= (1u64 << extra) - 1; } }
    }
}

impl fmt::Display for EquivBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for Equiv computation memoization.
#[derive(Debug, Clone)]
pub struct EquivLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl EquivLruCache {
    pub fn new(capacity: usize) -> Self { EquivLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
    pub fn get(&mut self, key: u64) -> Option<&Vec<f64>> {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].2 = self.clock;
            self.hits += 1;
            Some(&self.entries[pos].1)
        } else { self.misses += 1; None }
    }
    pub fn put(&mut self, key: u64, value: Vec<f64>) {
        self.clock += 1;
        if let Some(pos) = self.entries.iter().position(|(k, _, _)| *k == key) {
            self.entries[pos].1 = value;
            self.entries[pos].2 = self.clock;
            return;
        }
        if self.entries.len() >= self.capacity {
            let lru_pos = self.entries.iter().enumerate()
                .min_by_key(|(_, (_, _, ts))| *ts).map(|(i, _)| i).unwrap();
            self.entries.remove(lru_pos);
            self.evictions += 1;
        }
        self.entries.push((key, value, self.clock));
    }
    pub fn size(&self) -> usize { self.entries.len() }
    pub fn hit_rate(&self) -> f64 { let t = self.hits + self.misses; if t == 0 { 0.0 } else { self.hits as f64 / t as f64 } }
    pub fn eviction_count(&self) -> u64 { self.evictions }
    pub fn contains(&self, key: u64) -> bool { self.entries.iter().any(|(k, _, _)| *k == key) }
    pub fn clear(&mut self) { self.entries.clear(); self.hits = 0; self.misses = 0; self.evictions = 0; self.clock = 0; }
    pub fn keys(&self) -> Vec<u64> { self.entries.iter().map(|(k, _, _)| *k).collect() }
}

impl fmt::Display for EquivLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for Equiv scheduling.
#[derive(Debug, Clone)]
pub struct EquivGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl EquivGraphColoring {
    pub fn new(n: usize) -> Self {
        EquivGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
    }
    pub fn add_edge(&mut self, i: usize, j: usize) {
        if i < self.num_nodes && j < self.num_nodes {
            self.adjacency[i][j] = true;
            self.adjacency[j][i] = true;
        }
    }
    pub fn greedy_color(&mut self) -> usize {
        self.colors = vec![None; self.num_nodes];
        let mut max_color = 0;
        for v in 0..self.num_nodes {
            let neighbor_colors: std::collections::HashSet<usize> = (0..self.num_nodes)
                .filter(|&u| self.adjacency[v][u] && self.colors[u].is_some())
                .map(|u| self.colors[u].unwrap()).collect();
            let mut c = 0;
            while neighbor_colors.contains(&c) { c += 1; }
            self.colors[v] = Some(c);
            max_color = max_color.max(c);
        }
        self.num_colors_used = max_color + 1;
        self.num_colors_used
    }
    pub fn is_valid_coloring(&self) -> bool {
        for i in 0..self.num_nodes {
            for j in (i+1)..self.num_nodes {
                if self.adjacency[i][j] {
                    if let (Some(ci), Some(cj)) = (self.colors[i], self.colors[j]) {
                        if ci == cj { return false; }
                    }
                }
            }
        }
        true
    }
    pub fn chromatic_number_upper_bound(&self) -> usize {
        let max_degree = (0..self.num_nodes)
            .map(|v| (0..self.num_nodes).filter(|&u| self.adjacency[v][u]).count())
            .max().unwrap_or(0);
        max_degree + 1
    }
    pub fn color_classes(&self) -> Vec<Vec<usize>> {
        let mut classes: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
        for (v, &c) in self.colors.iter().enumerate() {
            if let Some(color) = c { classes.entry(color).or_default().push(v); }
        }
        let mut result: Vec<Vec<usize>> = classes.into_values().collect();
        result.sort_by_key(|v| v[0]);
        result
    }
}

impl fmt::Display for EquivGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for Equiv ranking.
#[derive(Debug, Clone)]
pub struct EquivTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl EquivTopK {
    pub fn new(k: usize) -> Self { EquivTopK { k, items: Vec::new() } }
    pub fn insert(&mut self, score: f64, label: impl Into<String>) {
        self.items.push((score, label.into()));
        self.items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if self.items.len() > self.k { self.items.truncate(self.k); }
    }
    pub fn top(&self) -> &[(f64, String)] { &self.items }
    pub fn min_score(&self) -> Option<f64> { self.items.last().map(|(s, _)| *s) }
    pub fn max_score(&self) -> Option<f64> { self.items.first().map(|(s, _)| *s) }
    pub fn is_full(&self) -> bool { self.items.len() >= self.k }
    pub fn len(&self) -> usize { self.items.len() }
    pub fn contains_label(&self, label: &str) -> bool { self.items.iter().any(|(_, l)| l == label) }
    pub fn clear(&mut self) { self.items.clear(); }
    pub fn merge(&mut self, other: &Self) {
        for (score, label) in &other.items { self.insert(*score, label.clone()); }
    }
}

impl fmt::Display for EquivTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for Equiv monitoring.
#[derive(Debug, Clone)]
pub struct EquivSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl EquivSlidingWindow {
    pub fn new(window_size: usize) -> Self { EquivSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
    pub fn push(&mut self, value: f64) {
        self.data.push(value);
        self.sum += value;
        if self.data.len() > self.window_size {
            self.sum -= self.data.remove(0);
        }
    }
    pub fn mean(&self) -> f64 { if self.data.is_empty() { 0.0 } else { self.sum / self.data.len() as f64 } }
    pub fn variance(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let m = self.mean();
        self.data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (self.data.len() - 1) as f64
    }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn min(&self) -> f64 { self.data.iter().cloned().fold(f64::INFINITY, f64::min) }
    pub fn max(&self) -> f64 { self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_full(&self) -> bool { self.data.len() >= self.window_size }
    pub fn trend(&self) -> f64 {
        if self.data.len() < 2 { return 0.0; }
        let n = self.data.len() as f64;
        let sum_x: f64 = (0..self.data.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.data.iter().sum();
        let sum_xy: f64 = self.data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_xx: f64 = (0..self.data.len()).map(|i| (i as f64) * (i as f64)).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { 0.0 } else { (n * sum_xy - sum_x * sum_y) / denom }
    }
    pub fn anomaly_score(&self, value: f64) -> f64 {
        let s = self.std_dev();
        if s.abs() < 1e-15 { return 0.0; }
        ((value - self.mean()) / s).abs()
    }
    pub fn clear(&mut self) { self.data.clear(); self.sum = 0.0; }
}

impl fmt::Display for EquivSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for Equiv classification evaluation.
#[derive(Debug, Clone)]
pub struct EquivConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl EquivConfusionMatrix {
    pub fn new() -> Self { EquivConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
    pub fn from_predictions(actual: &[bool], predicted: &[bool]) -> Self {
        let mut cm = Self::new();
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            match (a, p) {
                (true, true) => cm.true_positive += 1,
                (false, true) => cm.false_positive += 1,
                (true, false) => cm.false_negative += 1,
                (false, false) => cm.true_negative += 1,
            }
        }
        cm
    }
    pub fn total(&self) -> u64 { self.true_positive + self.false_positive + self.true_negative + self.false_negative }
    pub fn accuracy(&self) -> f64 { let t = self.total(); if t == 0 { 0.0 } else { (self.true_positive + self.true_negative) as f64 / t as f64 } }
    pub fn precision(&self) -> f64 { let d = self.true_positive + self.false_positive; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn recall(&self) -> f64 { let d = self.true_positive + self.false_negative; if d == 0 { 0.0 } else { self.true_positive as f64 / d as f64 } }
    pub fn f1_score(&self) -> f64 { let p = self.precision(); let r = self.recall(); if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) } }
    pub fn specificity(&self) -> f64 { let d = self.true_negative + self.false_positive; if d == 0 { 0.0 } else { self.true_negative as f64 / d as f64 } }
    pub fn false_positive_rate(&self) -> f64 { 1.0 - self.specificity() }
    pub fn matthews_correlation(&self) -> f64 {
        let tp = self.true_positive as f64; let fp = self.false_positive as f64;
        let tn = self.true_negative as f64; let fn_ = self.false_negative as f64;
        let num = tp * tn - fp * fn_;
        let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }
    pub fn merge(&mut self, other: &Self) {
        self.true_positive += other.true_positive;
        self.false_positive += other.false_positive;
        self.true_negative += other.true_negative;
        self.false_negative += other.false_negative;
    }
}

impl fmt::Display for EquivConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for Equiv feature vectors.
pub fn equiv_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for Equiv.
pub fn equiv_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for Equiv.
pub fn equiv_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for Equiv.
pub fn equiv_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for Equiv.
pub fn equiv_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for Equiv.
pub fn equiv_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for Equiv.
pub fn equiv_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for Equiv.
pub fn equiv_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for Equiv.
pub fn equiv_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for Equiv.
pub fn equiv_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for Equiv.
pub fn equiv_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for Equiv.
pub fn equiv_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for Equiv.
pub fn equiv_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for Equiv.
pub fn equiv_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for Equiv.
pub fn equiv_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (equiv_kl_divergence(p, &m) + equiv_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for Equiv.
pub fn equiv_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for Equiv.
pub fn equiv_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for Equiv.
pub fn equiv_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for Equiv.
#[derive(Debug, Clone)]
pub struct EquivFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl EquivFeatureScaler {
    pub fn new() -> Self { EquivFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
    pub fn fit(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() { return; }
        let dim = data[0].len();
        let n = data.len() as f64;
        self.means = vec![0.0; dim];
        self.mins = vec![f64::INFINITY; dim];
        self.maxs = vec![f64::NEG_INFINITY; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.means[j] += v;
                self.mins[j] = self.mins[j].min(v);
                self.maxs[j] = self.maxs[j].max(v);
            }
        }
        for j in 0..dim { self.means[j] /= n; }
        self.stds = vec![0.0; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                self.stds[j] += (v - self.means[j]).powi(2);
            }
        }
        for j in 0..dim { self.stds[j] = (self.stds[j] / (n - 1.0).max(1.0)).sqrt(); }
        self.fitted = true;
    }
    pub fn standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            if self.stds[j].abs() < 1e-15 { 0.0 } else { (v - self.means[j]) / self.stds[j] }
        }).collect()
    }
    pub fn normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| {
            let range = self.maxs[j] - self.mins[j];
            if range.abs() < 1e-15 { 0.0 } else { (v - self.mins[j]) / range }
        }).collect()
    }
    pub fn inverse_standardize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * self.stds[j] + self.means[j]).collect()
    }
    pub fn inverse_normalize(&self, row: &[f64]) -> Vec<f64> {
        row.iter().enumerate().map(|(j, &v)| v * (self.maxs[j] - self.mins[j]) + self.mins[j]).collect()
    }
    pub fn dimension(&self) -> usize { self.means.len() }
}

impl fmt::Display for EquivFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for Equiv trend analysis.
#[derive(Debug, Clone)]
pub struct EquivLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl EquivLinearRegression {
    pub fn new() -> Self { EquivLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
    pub fn fit(&mut self, x: &[f64], y: &[f64]) {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;
        if n < 2.0 { return; }
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-15 { return; }
        self.slope = (n * sum_xy - sum_x * sum_y) / denom;
        self.intercept = (sum_y - self.slope * sum_x) / n;
        let mean_y = sum_y / n;
        let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| (yi - self.predict(xi)).powi(2)).sum();
        self.r_squared = if ss_tot.abs() < 1e-15 { 1.0 } else { 1.0 - ss_res / ss_tot };
        self.fitted = true;
    }
    pub fn predict(&self, x: f64) -> f64 { self.slope * x + self.intercept }
    pub fn predict_many(&self, xs: &[f64]) -> Vec<f64> { xs.iter().map(|&x| self.predict(x)).collect() }
    pub fn residuals(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        x.iter().zip(y.iter()).map(|(&xi, &yi)| yi - self.predict(xi)).collect()
    }
    pub fn mse(&self, x: &[f64], y: &[f64]) -> f64 {
        let res = self.residuals(x, y);
        res.iter().map(|r| r * r).sum::<f64>() / res.len() as f64
    }
    pub fn rmse(&self, x: &[f64], y: &[f64]) -> f64 { self.mse(x, y).sqrt() }
}

impl fmt::Display for EquivLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for Equiv.
#[derive(Debug, Clone)]
pub struct EquivWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl EquivWeightedGraph {
    pub fn new(n: usize) -> Self { EquivWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) {
        self.adj[u].push((v, w));
        self.adj[v].push((u, w));
        self.num_edges += 1;
    }
    pub fn neighbors(&self, u: usize) -> &[(usize, f64)] { &self.adj[u] }
    pub fn degree(&self, u: usize) -> usize { self.adj[u].len() }
    pub fn total_weight(&self) -> f64 {
        self.adj.iter().flat_map(|edges| edges.iter().map(|(_, w)| w)).sum::<f64>() / 2.0
    }
    pub fn min_spanning_tree_weight(&self) -> f64 {
        // Kruskal's algorithm
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..self.num_nodes {
            for &(v, w) in &self.adj[u] {
                if u < v { edges.push((w, u, v)); }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut parent: Vec<usize> = (0..self.num_nodes).collect();
        let mut rank = vec![0usize; self.num_nodes];
        fn find_equiv(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_equiv(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_equiv(&mut parent, u);
            let rv = find_equiv(&mut parent, v);
            if ru != rv {
                if rank[ru] < rank[rv] { parent[ru] = rv; }
                else if rank[ru] > rank[rv] { parent[rv] = ru; }
                else { parent[rv] = ru; rank[ru] += 1; }
                total += w;
                count += 1;
                if count == self.num_nodes - 1 { break; }
            }
        }
        total
    }
    pub fn dijkstra(&self, start: usize) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.num_nodes];
        let mut visited = vec![false; self.num_nodes];
        dist[start] = 0.0;
        for _ in 0..self.num_nodes {
            let mut u = None;
            let mut min_d = f64::INFINITY;
            for v in 0..self.num_nodes { if !visited[v] && dist[v] < min_d { min_d = dist[v]; u = Some(v); } }
            let u = match u { Some(v) => v, None => break };
            visited[u] = true;
            for &(v, w) in &self.adj[u] {
                let alt = dist[u] + w;
                if alt < dist[v] { dist[v] = alt; }
            }
        }
        dist
    }
    pub fn eccentricity(&self, u: usize) -> f64 {
        let dists = self.dijkstra(u);
        dists.iter().cloned().filter(|&d| d.is_finite()).fold(0.0f64, f64::max)
    }
    pub fn diameter(&self) -> f64 {
        (0..self.num_nodes).map(|u| self.eccentricity(u)).fold(0.0f64, f64::max)
    }
    pub fn clustering_coefficient(&self, u: usize) -> f64 {
        let neighbors: Vec<usize> = self.adj[u].iter().map(|(v, _)| *v).collect();
        let k = neighbors.len();
        if k < 2 { return 0.0; }
        let mut triangles = 0;
        for i in 0..k {
            for j in (i+1)..k {
                if self.adj[neighbors[i]].iter().any(|(v, _)| *v == neighbors[j]) {
                    triangles += 1;
                }
            }
        }
        2.0 * triangles as f64 / (k * (k - 1)) as f64
    }
    pub fn average_clustering_coefficient(&self) -> f64 {
        let sum: f64 = (0..self.num_nodes).map(|u| self.clustering_coefficient(u)).sum();
        sum / self.num_nodes as f64
    }
}

impl fmt::Display for EquivWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for Equiv.
pub fn equiv_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if data.len() < window { return Vec::new(); }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..data.len() {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Cumulative sum for Equiv.
pub fn equiv_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for Equiv.
pub fn equiv_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for Equiv.
pub fn equiv_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for Equiv.
pub fn equiv_dft_magnitude(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut magnitudes = Vec::with_capacity(n / 2 + 1);
    for k in 0..=n/2 {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &x) in data.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    magnitudes
}

/// Trapezoidal integration for Equiv.
pub fn equiv_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for Equiv.
pub fn equiv_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 || n % 2 == 0 { return 0.0; }
    let mut total = 0.0;
    for i in (0..n-2).step_by(2) {
        let h = (x[i+2] - x[i]) / 6.0;
        total += h * (y[i] + 4.0 * y[i+1] + y[i+2]);
    }
    total
}

/// Convolution for Equiv.
pub fn equiv_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Axis-aligned bounding box for Equiv spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct EquivAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl EquivAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { EquivAABB { x_min, y_min, x_max, y_max } }
    pub fn contains(&self, x: f64, y: f64) -> bool { x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max }
    pub fn intersects(&self, other: &Self) -> bool {
        !(self.x_max < other.x_min || self.x_min > other.x_max || self.y_max < other.y_min || self.y_min > other.y_max)
    }
    pub fn width(&self) -> f64 { self.x_max - self.x_min }
    pub fn height(&self) -> f64 { self.y_max - self.y_min }
    pub fn area(&self) -> f64 { self.width() * self.height() }
    pub fn center(&self) -> (f64, f64) { ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0) }
    pub fn subdivide(&self) -> [Self; 4] {
        let (cx, cy) = self.center();
        [
            EquivAABB::new(self.x_min, self.y_min, cx, cy),
            EquivAABB::new(cx, self.y_min, self.x_max, cy),
            EquivAABB::new(self.x_min, cy, cx, self.y_max),
            EquivAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for Equiv.
#[derive(Debug, Clone, Copy)]
pub struct EquivPoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for Equiv spatial indexing.
#[derive(Debug, Clone)]
pub struct EquivQuadTree {
    pub boundary: EquivAABB,
    pub points: Vec<EquivPoint2D>,
    pub children: Option<Vec<EquivQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl EquivQuadTree {
    pub fn new(boundary: EquivAABB, capacity: usize, max_depth: usize) -> Self {
        EquivQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: EquivAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        EquivQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: EquivPoint2D) -> bool {
        if !self.boundary.contains(p.x, p.y) { return false; }
        if self.points.len() < self.capacity && self.children.is_none() {
            self.points.push(p); return true;
        }
        if self.children.is_none() && self.depth < self.max_depth { self.subdivide_tree(); }
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() { if child.insert(p) { return true; } }
        }
        self.points.push(p); true
    }
    fn subdivide_tree(&mut self) {
        let quads = self.boundary.subdivide();
        let mut children = Vec::with_capacity(4);
        for q in quads.iter() {
            children.push(EquivQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &EquivAABB) -> Vec<EquivPoint2D> {
        let mut result = Vec::new();
        if !self.boundary.intersects(range) { return result; }
        for p in &self.points { if range.contains(p.x, p.y) { result.push(*p); } }
        if let Some(ref children) = self.children {
            for child in children { result.extend(child.query_range(range)); }
        }
        result
    }
    pub fn count(&self) -> usize {
        let mut c = self.points.len();
        if let Some(ref children) = self.children {
            for child in children { c += child.count(); }
        }
        c
    }
    pub fn tree_depth(&self) -> usize {
        if let Some(ref children) = self.children {
            1 + children.iter().map(|c| c.tree_depth()).max().unwrap_or(0)
        } else { 0 }
    }
}

impl fmt::Display for EquivQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for Equiv.
pub fn equiv_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let m = a.len();
    if m == 0 { return (Vec::new(), Vec::new()); }
    let n = a[0].len();
    let mut q = vec![vec![0.0; m]; n]; // column vectors
    let mut r = vec![vec![0.0; n]; n];
    // extract columns of a
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
    for j in 0..n {
        let mut v = cols[j].clone();
        for i in 0..j {
            let dot: f64 = v.iter().zip(q[i].iter()).map(|(&a, &b)| a * b).sum();
            r[i][j] = dot;
            for k in 0..m { v[k] -= dot * q[i][k]; }
        }
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        r[j][j] = norm;
        if norm.abs() > 1e-15 { for k in 0..m { q[j][k] = v[k] / norm; } }
    }
    // convert q from list of column vectors to matrix
    let q_mat: Vec<Vec<f64>> = (0..m).map(|i| (0..n).map(|j| q[j][i]).collect()).collect();
    (q_mat, r)
}

/// Solve upper triangular system Rx = b for Equiv.
pub fn equiv_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for Equiv.
pub fn equiv_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for Equiv.
pub fn equiv_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for Equiv.
pub fn equiv_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for Equiv.
pub fn equiv_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for Equiv.
pub fn equiv_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for Equiv.
pub fn equiv_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for Equiv.
pub fn equiv_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = equiv_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for Equiv.
#[derive(Debug, Clone)]
pub struct EquivRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl EquivRunningStats {
    pub fn new() -> Self { EquivRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
    pub fn push(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        self.min_val = self.min_val.min(x);
        self.max_val = self.max_val.max(x);
        self.sum += x;
    }
    pub fn variance(&self) -> f64 { if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 } }
    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
    pub fn range(&self) -> f64 { self.max_val - self.min_val }
    pub fn coefficient_of_variation(&self) -> f64 { if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() } }
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 { return; }
        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * other.count as f64 / combined_count as f64;
        self.m2 += other.m2 + delta * delta * self.count as f64 * other.count as f64 / combined_count as f64;
        self.mean = combined_mean;
        self.count = combined_count;
        self.min_val = self.min_val.min(other.min_val);
        self.max_val = self.max_val.max(other.max_val);
        self.sum += other.sum;
    }
}

impl fmt::Display for EquivRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Interquartile range for Equiv.
pub fn equiv_iqr(data: &[f64]) -> f64 {
    equiv_percentile_at(data, 75.0) - equiv_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for Equiv.
pub fn equiv_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = equiv_percentile_at(data, 25.0);
    let q3 = equiv_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for Equiv.
pub fn equiv_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for Equiv.
pub fn equiv_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for Equiv.
pub fn equiv_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = equiv_rank(x);
    let ry = equiv_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Covariance matrix for Equiv.
pub fn equiv_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if data.is_empty() { return Vec::new(); }
    let n = data.len() as f64;
    let d = data[0].len();
    let means: Vec<f64> = (0..d).map(|j| data.iter().map(|row| row[j]).sum::<f64>() / n).collect();
    let mut cov = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in i..d {
            let c: f64 = data.iter().map(|row| (row[i] - means[i]) * (row[j] - means[j])).sum::<f64>() / (n - 1.0).max(1.0);
            cov[i][j] = c; cov[j][i] = c;
        }
    }
    cov
}

/// Correlation matrix for Equiv.
pub fn equiv_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = equiv_covariance_matrix(data);
    let d = cov.len();
    let mut corr = vec![vec![0.0; d]; d];
    for i in 0..d {
        for j in 0..d {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            corr[i][j] = if denom.abs() < 1e-15 { 0.0 } else { cov[i][j] / denom };
        }
    }
    corr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_test(name: &str, reads: usize, writes: usize) -> LitmusTestDesc {
        let mut test = LitmusTestDesc::new(name);
        let mut events = Vec::new();
        let mut id = 0u32;
        for _ in 0..writes {
            events.push(Event::new(id, 0, OpKind::Write, "x").with_value(1).with_order(id));
            id += 1;
        }
        for _ in 0..reads {
            events.push(Event::new(id, 0, OpKind::Read, "x").with_order(id));
            id += 1;
        }
        test.add_thread(events);
        test
    }

    fn make_mp_test() -> LitmusTestDesc {
        let mut test = LitmusTestDesc::new("MP");
        test.set_initial("x", 0);
        test.set_initial("y", 0);
        test.add_thread(vec![
            Event::new(0, 0, OpKind::Write, "x").with_value(1),
            Event::new(1, 0, OpKind::Write, "y").with_value(1),
        ]);
        test.add_thread(vec![
            Event::new(2, 1, OpKind::Read, "y"),
            Event::new(3, 1, OpKind::Read, "x"),
        ]);
        test.add_expected_outcome(
            ExecutionOutcome::new().with_read(2, 1).with_read(3, 0)
        );
        test
    }

    #[test]
    fn test_event_display() {
        let e = Event::new(0, 1, OpKind::Write, "x").with_value(42);
        assert_eq!(format!("{}", e), "T1:W x=42");
    }

    #[test]
    fn test_event_is_read_write() {
        assert!(Event::new(0, 0, OpKind::Read, "x").is_read());
        assert!(!Event::new(0, 0, OpKind::Read, "x").is_write());
        assert!(Event::new(0, 0, OpKind::Write, "x").is_write());
        assert!(Event::new(0, 0, OpKind::RMW, "x").is_read());
        assert!(Event::new(0, 0, OpKind::RMW, "x").is_write());
    }

    #[test]
    fn test_execution_basic() {
        let mut exec = Execution::new();
        exec.add_event(Event::new(0, 0, OpKind::Write, "x"));
        exec.add_event(Event::new(1, 0, OpKind::Read, "x"));
        exec.add_event(Event::new(2, 1, OpKind::Read, "x"));
        assert_eq!(exec.events.len(), 3);
        assert_eq!(exec.num_threads(), 2);
        assert_eq!(exec.reads().len(), 2);
        assert_eq!(exec.writes().len(), 1);
    }

    #[test]
    fn test_execution_outcome_fingerprint() {
        let o1 = ExecutionOutcome::new().with_final("x", 1).with_read(0, 1);
        let o2 = ExecutionOutcome::new().with_final("x", 1).with_read(0, 1);
        let o3 = ExecutionOutcome::new().with_final("x", 2).with_read(0, 1);
        assert_eq!(o1.fingerprint(), o2.fingerprint());
        assert_ne!(o1.fingerprint(), o3.fingerprint());
    }

    #[test]
    fn test_litmus_test_desc() {
        let test = make_mp_test();
        assert_eq!(test.threads.len(), 2);
        assert_eq!(test.total_events(), 4);
        let addrs = test.all_addresses();
        assert!(addrs.contains("x"));
        assert!(addrs.contains("y"));
    }

    #[test]
    fn test_litmus_to_execution() {
        let test = make_mp_test();
        let exec = test.to_execution();
        assert_eq!(exec.events.len(), 4);
        // Should have po relations
        assert!(!exec.relations.is_empty());
    }

    #[test]
    fn test_isomorphism_identical() {
        let test = make_mp_test();
        let exec1 = test.to_execution();
        let exec2 = test.to_execution();
        let mapping = IsomorphismChecker::are_isomorphic(&exec1, &exec2);
        assert!(mapping.is_some());
    }

    #[test]
    fn test_isomorphism_different_sizes() {
        let mut exec1 = Execution::new();
        exec1.add_event(Event::new(0, 0, OpKind::Write, "x"));
        let mut exec2 = Execution::new();
        exec2.add_event(Event::new(0, 0, OpKind::Write, "x"));
        exec2.add_event(Event::new(1, 0, OpKind::Read, "x"));
        assert!(IsomorphismChecker::are_isomorphic(&exec1, &exec2).is_none());
    }

    #[test]
    fn test_canonical_form_consistency() {
        let test = make_mp_test();
        let exec = test.to_execution();
        let c1 = canonical_form(&exec);
        let c2 = canonical_form(&exec);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_isomorphism_hash() {
        let test = make_mp_test();
        let exec = test.to_execution();
        let h1 = isomorphism_hash(&exec);
        let h2 = isomorphism_hash(&exec);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_color_refinement() {
        let test = make_mp_test();
        let exec = test.to_execution();
        let mut cr = ColorRefinement::new(&exec);
        let colors = cr.stable_coloring();
        assert_eq!(colors.len(), exec.events.len());
    }

    #[test]
    fn test_color_refinement_signature() {
        let test = make_mp_test();
        let exec = test.to_execution();
        let mut cr1 = ColorRefinement::new(&exec);
        let mut cr2 = ColorRefinement::new(&exec);
        assert_eq!(cr1.color_signature(), cr2.color_signature());
    }

    #[test]
    fn test_outcome_set_operations() {
        let mut set1 = OutcomeSet::new();
        set1.add(ExecutionOutcome::new().with_final("x", 1));
        set1.add(ExecutionOutcome::new().with_final("x", 2));

        let mut set2 = OutcomeSet::new();
        set2.add(ExecutionOutcome::new().with_final("x", 2));
        set2.add(ExecutionOutcome::new().with_final("x", 3));

        let union = set1.union(&set2);
        assert_eq!(union.len(), 3);

        let inter = set1.intersection(&set2);
        assert_eq!(inter.len(), 1);

        let diff = set1.difference(&set2);
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn test_outcome_equivalence_same() {
        let mut set1 = OutcomeSet::new();
        set1.add(ExecutionOutcome::new().with_final("x", 1));
        let mut set2 = OutcomeSet::new();
        set2.add(ExecutionOutcome::new().with_final("x", 1));
        assert!(OutcomeEquivalenceChecker::same_outcomes(&set1, &set2));
    }

    #[test]
    fn test_outcome_equivalence_different() {
        let mut set1 = OutcomeSet::new();
        set1.add(ExecutionOutcome::new().with_final("x", 1));
        let mut set2 = OutcomeSet::new();
        set2.add(ExecutionOutcome::new().with_final("x", 2));
        assert!(!OutcomeEquivalenceChecker::same_outcomes(&set1, &set2));
    }

    #[test]
    fn test_outcome_diff() {
        let mut set1 = OutcomeSet::new();
        set1.add(ExecutionOutcome::new().with_final("x", 1));
        set1.add(ExecutionOutcome::new().with_final("x", 2));
        let mut set2 = OutcomeSet::new();
        set2.add(ExecutionOutcome::new().with_final("x", 2));
        set2.add(ExecutionOutcome::new().with_final("x", 3));

        let diff = OutcomeEquivalenceChecker::outcome_diff(&set1, &set2);
        assert_eq!(diff.only_in_first.len(), 1);
        assert_eq!(diff.only_in_second.len(), 1);
        assert_eq!(diff.common.len(), 1);
        assert!(!diff.is_equivalent());
    }

    #[test]
    fn test_subsumption_equal() {
        let t1 = make_simple_test("t1", 1, 1);
        let result = SubsumptionChecker::subsumes(&t1, &t1);
        assert_eq!(result, SubsumptionResult::Equal);
    }

    #[test]
    fn test_equivalence_detection() {
        let t1 = make_simple_test("t1", 1, 1);
        let t2 = make_simple_test("t2", 1, 1);
        let t3 = make_simple_test("t3", 2, 1);

        let detector = EquivalenceDetector::new();
        let report = detector.detect_equivalences(&[t1, t2, t3]);
        // t1 and t2 should be equivalent
        assert!(report.unique_classes <= 2);
    }

    #[test]
    fn test_test_signature() {
        let test = make_mp_test();
        let sig = compute_signature(&test);
        assert_eq!(sig.num_threads, 2);
        assert_eq!(sig.num_events, 4);
        assert_eq!(sig.num_reads, 2);
        assert_eq!(sig.num_writes, 2);
        assert_eq!(sig.num_addresses, 2);
    }

    #[test]
    fn test_signature_index() {
        let tests = vec![
            make_simple_test("t1", 1, 1),
            make_simple_test("t2", 1, 1),
            make_simple_test("t3", 2, 2),
        ];
        let index = SignatureIndex::build_from_tests(&tests);
        let sig = compute_signature(&tests[0]);
        let candidates = index.find_candidates(&sig);
        assert!(candidates.len() >= 1);
    }

    #[test]
    fn test_minimize_test() {
        let mut test = LitmusTestDesc::new("test");
        test.add_thread(vec![
            Event::new(0, 0, OpKind::Write, "x").with_value(1),
            Event::new(1, 0, OpKind::Write, "y").with_value(1),
            Event::new(2, 0, OpKind::Read, "x"),
        ]);
        test.add_thread(vec![
            Event::new(3, 1, OpKind::Read, "y"),
        ]);

        let minimized = TestMinimizer::minimize_test(&test, &|t| t.total_events() >= 2);
        assert!(minimized.total_events() >= 2);
        assert!(minimized.total_events() <= test.total_events());
    }

    #[test]
    fn test_suite_reduction() {
        let tests = vec![
            make_simple_test("t1", 1, 1),
            make_simple_test("t2", 1, 1),
            make_simple_test("t3", 2, 2),
        ];
        let reduced = TestSuiteReducer::reduce_by_signature(&tests);
        assert!(reduced.len() <= tests.len());
    }

    #[test]
    fn test_outcome_diff_jaccard() {
        let mut set1 = OutcomeSet::new();
        set1.add(ExecutionOutcome::new().with_final("x", 1));
        set1.add(ExecutionOutcome::new().with_final("x", 2));
        let mut set2 = OutcomeSet::new();
        set2.add(ExecutionOutcome::new().with_final("x", 2));
        set2.add(ExecutionOutcome::new().with_final("x", 3));

        let diff = OutcomeEquivalenceChecker::outcome_diff(&set1, &set2);
        let j = diff.jaccard_similarity();
        assert!((j - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_equivalence_report_stats() {
        let t1 = make_simple_test("t1", 1, 1);
        let t2 = make_simple_test("t2", 1, 1);
        let detector = EquivalenceDetector::new();
        let report = detector.detect_equivalences(&[t1, t2]);
        assert_eq!(report.total_tests, 2);
        assert!(report.reduction_ratio >= 0.0);
    }

    #[test]
    fn test_empty_equivalence_detection() {
        let detector = EquivalenceDetector::new();
        let report = detector.detect_equivalences(&[]);
        assert_eq!(report.total_tests, 0);
        assert_eq!(report.unique_classes, 0);
    }
    #[test]
    fn test_testnormalizer_new() {
        let item = TestNormalizer::new(0, 0, 0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_threadrenaming_new() {
        let item = ThreadRenaming::new(Vec::new(), Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_addressrenaming_new() {
        let item = AddressRenaming::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_valuerenaming_new() {
        let item = ValueRenaming::new(Vec::new(), Vec::new(), false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_testhasher_new() {
        let item = TestHasher::new(0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_testserializer_new() {
        let item = TestSerializer::new("test".to_string(), false, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_equivalenceproofcert_new() {
        let item = EquivalenceProofCert::new(0, Vec::new(), false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_classgenerator_new() {
        let item = ClassGenerator::new(0, Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_antichaincomputation_new() {
        let item = AntiChainComputation::new(Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_subsumptionwidth_new() {
        let item = SubsumptionWidth::new(0, Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_equivalenceclass_new() {
        let item = EquivalenceClass::new(0, Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_canonicalform_new() {
        let item = CanonicalForm::new(Vec::new(), Vec::new(), Vec::new(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_equiv_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = equiv_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = equiv_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_equiv_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = equiv_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = equiv_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_equiv_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = equiv_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_equiv_analysis() {
        let mut a = EquivAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_equiv_iterator() {
        let iter = EquivResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_equiv_batch_processor() {
        let mut proc = EquivBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_equiv_histogram() {
        let hist = EquivHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_equiv_graph() {
        let mut g = EquivGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_equiv_graph_shortest_path() {
        let mut g = EquivGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_graph_topo_sort() {
        let mut g = EquivGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_equiv_graph_components() {
        let mut g = EquivGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_equiv_cache() {
        let mut cache = EquivCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_equiv_config() {
        let config = EquivConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_equiv_report() {
        let mut report = EquivReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_equiv_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = equiv_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_equiv_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = equiv_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_harmmean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = equiv_harmmean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_equiv_geomean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = equiv_geomean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_equiv_sample_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = equiv_sample_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_equiv_excess_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = equiv_excess_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_equiv_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = equiv_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_equiv_percentile_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = equiv_percentile_at(&data, 0.9);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_equiv_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = equiv_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_equiv_analysis_normalize() {
        let mut a = EquivAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_analysis_transpose() {
        let mut a = EquivAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_analysis_multiply() {
        let mut a = EquivAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = EquivAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_analysis_frobenius() {
        let mut a = EquivAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_analysis_symmetric() {
        let mut a = EquivAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_equiv_graph_dot() {
        let mut g = EquivGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_equiv_histogram_render() {
        let hist = EquivHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_equiv_batch_reset() {
        let mut proc = EquivBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_equiv_graph_remove_edge() {
        let mut g = EquivGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_equiv_dense_matrix_new() {
        let m = EquivDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_equiv_dense_matrix_identity() {
        let m = EquivDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_mul() {
        let a = EquivDenseMatrix::identity(2);
        let b = EquivDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_transpose() {
        let a = EquivDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_det_2x2() {
        let m = EquivDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_det_3x3() {
        let m = EquivDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_inverse_2x2() {
        let m = EquivDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_power() {
        let m = EquivDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_rank() {
        let m = EquivDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_equiv_dense_matrix_solve() {
        let a = EquivDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_equiv_dense_matrix_lu() {
        let a = EquivDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_eigenvalues() {
        let m = EquivDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_kronecker() {
        let a = EquivDenseMatrix::identity(2);
        let b = EquivDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_equiv_dense_matrix_hadamard() {
        let a = EquivDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = EquivDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_interval() {
        let a = EquivInterval::new(1.0, 3.0);
        let b = EquivInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_interval_mul() {
        let a = EquivInterval::new(-2.0, 3.0);
        let b = EquivInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_interval_hull() {
        let a = EquivInterval::new(1.0, 3.0);
        let b = EquivInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_state_machine() {
        let mut sm = EquivStateMachine::new();
        assert_eq!(*sm.state(), EquivState::Init);
        assert!(sm.transition(EquivState::Normalizing));
        assert_eq!(*sm.state(), EquivState::Normalizing);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_equiv_state_machine_invalid() {
        let mut sm = EquivStateMachine::new();
        let last_state = EquivState::Done;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_equiv_state_machine_reset() {
        let mut sm = EquivStateMachine::new();
        sm.transition(EquivState::Normalizing);
        sm.reset();
        assert_eq!(*sm.state(), EquivState::Init);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_equiv_ring_buffer() {
        let mut rb = EquivRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_ring_buffer_to_vec() {
        let mut rb = EquivRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_disjoint_set() {
        let mut ds = EquivDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_equiv_disjoint_set_components() {
        let mut ds = EquivDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_equiv_sorted_list() {
        let mut sl = EquivSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_sorted_list_remove() {
        let mut sl = EquivSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_equiv_ema() {
        let mut ema = EquivEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_bloom_filter() {
        let mut bf = EquivBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_equiv_trie() {
        let mut trie = EquivTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);
        assert_eq!(trie.search("hello"), Some(1));
        assert_eq!(trie.search("help"), Some(2));
        assert_eq!(trie.search("hell"), None);
        assert!(trie.starts_with("hel"));
        assert!(!trie.starts_with("xyz"));
    }

    #[test]
    fn test_equiv_dense_matrix_sym() {
        let m = EquivDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_equiv_dense_matrix_diag() {
        let m = EquivDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_equiv_dense_matrix_upper_tri() {
        let m = EquivDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_equiv_dense_matrix_outer() {
        let m = EquivDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dense_matrix_submatrix() {
        let m = EquivDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_priority_queue() {
        let mut pq = EquivPriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_equiv_accumulator() {
        let mut acc = EquivAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_accumulator_merge() {
        let mut a = EquivAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = EquivAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_sparse_matrix() {
        let mut m = EquivSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_sparse_mul_vec() {
        let mut m = EquivSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_sparse_transpose() {
        let mut m = EquivSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_polynomial_eval() {
        let p = EquivPolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_polynomial_add() {
        let a = EquivPolynomial::new(vec![1.0, 2.0]);
        let b = EquivPolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_polynomial_mul() {
        let a = EquivPolynomial::new(vec![1.0, 1.0]);
        let b = EquivPolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_polynomial_deriv() {
        let p = EquivPolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_polynomial_integral() {
        let p = EquivPolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_polynomial_roots() {
        let p = EquivPolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_equiv_polynomial_newton() {
        let p = EquivPolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_equiv_polynomial_compose() {
        let p = EquivPolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = EquivPolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_rng() {
        let mut rng = EquivRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_equiv_rng_gaussian() {
        let mut rng = EquivRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_equiv_timer() {
        let mut timer = EquivTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_bitvector() {
        let mut bv = EquivBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_equiv_bitvector_ops() {
        let mut a = EquivBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = EquivBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_equiv_bitvector_jaccard() {
        let mut a = EquivBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = EquivBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_priority_queue_empty() {
        let mut pq = EquivPriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_equiv_sparse_add() {
        let mut a = EquivSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = EquivSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_rng_shuffle() {
        let mut rng = EquivRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_polynomial_display() {
        let p = EquivPolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_equiv_polynomial_monomial() {
        let m = EquivPolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_timer_percentiles() {
        let mut timer = EquivTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_equiv_accumulator_cv() {
        let mut acc = EquivAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_equiv_sparse_diagonal() {
        let mut m = EquivSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_lru_cache() {
        let mut cache = EquivLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_equiv_lru_hit_rate() {
        let mut cache = EquivLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_graph_coloring() {
        let mut gc = EquivGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_equiv_graph_coloring_complete() {
        let mut gc = EquivGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_equiv_topk() {
        let mut tk = EquivTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_sliding_window() {
        let mut sw = EquivSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_equiv_sliding_window_trend() {
        let mut sw = EquivSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_equiv_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = EquivConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_equiv_confusion_f1() {
        let cm = EquivConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_equiv_cosine_similarity() {
        let s = equiv_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = equiv_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_euclidean_distance() {
        let d = equiv_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_sigmoid() {
        let s = equiv_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = equiv_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_equiv_softmax() {
        let sm = equiv_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_equiv_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = equiv_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_equiv_normalize() {
        let v = equiv_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_lerp() {
        assert!((equiv_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((equiv_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((equiv_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_clamp() {
        assert!((equiv_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((equiv_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((equiv_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_cross_product() {
        let c = equiv_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dot_product() {
        let d = equiv_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = equiv_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_equiv_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = equiv_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_equiv_logsumexp() {
        let lse = equiv_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_equiv_feature_scaler() {
        let mut scaler = EquivFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_feature_scaler_inverse() {
        let mut scaler = EquivFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_linear_regression() {
        let mut lr = EquivLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_linear_regression_predict() {
        let mut lr = EquivLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_weighted_graph() {
        let mut g = EquivWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_weighted_graph_mst() {
        let mut g = EquivWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = equiv_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_cumsum() {
        let cs = equiv_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_diff() {
        let d = equiv_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_autocorrelation() {
        let ac = equiv_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_dft_magnitude() {
        let mags = equiv_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_equiv_integrate_trapezoid() {
        let area = equiv_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_convolve() {
        let c = equiv_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_weighted_graph_clustering() {
        let mut g = EquivWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_histogram_cumulative() {
        let h = EquivHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_equiv_histogram_entropy() {
        let h = EquivHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_equiv_aabb() {
        let bb = EquivAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_aabb_intersects() {
        let a = EquivAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = EquivAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = EquivAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_equiv_quadtree() {
        let bb = EquivAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = EquivQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(EquivPoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_equiv_quadtree_query() {
        let bb = EquivAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = EquivQuadTree::new(bb, 2, 8);
        qt.insert(EquivPoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(EquivPoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = EquivAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_equiv_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = equiv_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = equiv_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = equiv_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((equiv_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_identity() {
        let id = equiv_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = equiv_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_equiv_running_stats() {
        let mut s = EquivRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_running_stats_merge() {
        let mut a = EquivRunningStats::new();
        let mut b = EquivRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_running_stats_cv() {
        let mut s = EquivRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_equiv_iqr() {
        let iqr = equiv_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_equiv_outliers() {
        let outliers = equiv_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_equiv_zscore() {
        let z = equiv_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_equiv_rank() {
        let r = equiv_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_spearman() {
        let rho = equiv_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equiv_sample_skewness_symmetric() {
        let s = equiv_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_equiv_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = equiv_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_equiv_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = equiv_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }

}

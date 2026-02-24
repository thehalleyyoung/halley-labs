#![allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Memory Model Specification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryModelSpec {
    pub name: String,
    pub axioms: Vec<AxiomSpec>,
    pub relations: Vec<RelationSpec>,
    pub constraints: Vec<ConstraintSpec>,
}

impl MemoryModelSpec {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            axioms: Vec::new(),
            relations: Vec::new(),
            constraints: Vec::new(),
        }
    }

    pub fn add_axiom(&mut self, axiom: AxiomSpec) {
        self.axioms.push(axiom);
    }

    pub fn add_relation(&mut self, relation: RelationSpec) {
        self.relations.push(relation);
    }

    pub fn add_constraint(&mut self, constraint: ConstraintSpec) {
        self.constraints.push(constraint);
    }

    pub fn axiom_names(&self) -> Vec<&str> {
        self.axioms.iter().map(|a| a.name.as_str()).collect()
    }

    pub fn relation_names(&self) -> Vec<&str> {
        self.relations.iter().map(|r| r.name.as_str()).collect()
    }

    pub fn find_axiom(&self, name: &str) -> Option<&AxiomSpec> {
        self.axioms.iter().find(|a| a.name == name)
    }

    pub fn find_relation(&self, name: &str) -> Option<&RelationSpec> {
        self.relations.iter().find(|r| r.name == name)
    }

    pub fn complexity_score(&self) -> usize {
        self.axioms.len() + self.relations.len() * 2 + self.constraints.len()
    }
}

impl fmt::Display for MemoryModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model({}, {} axioms, {} relations, {} constraints)",
            self.name, self.axioms.len(), self.relations.len(), self.constraints.len())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AxiomSpec {
    pub name: String,
    pub description: String,
    pub kind: AxiomKind,
    pub relations_used: Vec<String>,
}

impl AxiomSpec {
    pub fn new(name: impl Into<String>, kind: AxiomKind) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            kind,
            relations_used: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_relations(mut self, rels: Vec<String>) -> Self {
        self.relations_used = rels;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AxiomKind {
    Acyclicity,
    Irreflexivity,
    Totality,
    Emptiness,
    Inclusion,
    Custom,
}

impl fmt::Display for AxiomKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Acyclicity => write!(f, "acyclicity"),
            Self::Irreflexivity => write!(f, "irreflexivity"),
            Self::Totality => write!(f, "totality"),
            Self::Emptiness => write!(f, "emptiness"),
            Self::Inclusion => write!(f, "inclusion"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

// ---------------------------------------------------------------------------
// Constraints & Relations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConstraintSpec {
    pub name: String,
    pub kind: ConstraintKind,
    pub description: String,
}

impl ConstraintSpec {
    pub fn new(name: impl Into<String>, kind: ConstraintKind) -> Self {
        Self { name: name.into(), kind, description: String::new() }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintKind {
    WellFormed,
    Coherence,
    Atomicity,
    Causality,
    Ordering,
    Visibility,
    Consistency,
}

impl fmt::Display for ConstraintKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WellFormed => write!(f, "well-formed"),
            Self::Coherence => write!(f, "coherence"),
            Self::Atomicity => write!(f, "atomicity"),
            Self::Causality => write!(f, "causality"),
            Self::Ordering => write!(f, "ordering"),
            Self::Visibility => write!(f, "visibility"),
            Self::Consistency => write!(f, "consistency"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RelationSpec {
    pub name: String,
    pub definition: RelationDef,
    pub description: String,
    pub is_derived: bool,
}

impl RelationSpec {
    pub fn base(name: impl Into<String>, def: RelationDef) -> Self {
        Self { name: name.into(), definition: def, description: String::new(), is_derived: false }
    }

    pub fn derived(name: impl Into<String>, def: RelationDef) -> Self {
        Self { name: name.into(), definition: def, description: String::new(), is_derived: true }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn dependencies(&self) -> Vec<String> {
        self.definition.dependencies()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationDef {
    Base,
    Union(Vec<String>),
    Intersection(Vec<String>),
    Sequence(Vec<String>),
    Inverse(String),
    TransitiveClosure(String),
    ReflexiveTransitiveClosure(String),
    Difference(String, String),
    DomainRestriction(String, String),
    RangeRestriction(String, String),
    Optional(String),
    Custom(String),
}

impl RelationDef {
    pub fn dependencies(&self) -> Vec<String> {
        match self {
            Self::Base => Vec::new(),
            Self::Union(rels) | Self::Intersection(rels) | Self::Sequence(rels) => rels.clone(),
            Self::Inverse(r) | Self::TransitiveClosure(r) |
            Self::ReflexiveTransitiveClosure(r) | Self::Optional(r) | Self::Custom(r) => vec![r.clone()],
            Self::Difference(a, b) | Self::DomainRestriction(a, b) |
            Self::RangeRestriction(a, b) => vec![a.clone(), b.clone()],
        }
    }
}

impl fmt::Display for RelationDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Base => write!(f, "base"),
            Self::Union(rels) => write!(f, "({})", rels.join(" | ")),
            Self::Intersection(rels) => write!(f, "({})", rels.join(" & ")),
            Self::Sequence(rels) => write!(f, "({})", rels.join(" ; ")),
            Self::Inverse(r) => write!(f, "{}^-1", r),
            Self::TransitiveClosure(r) => write!(f, "{}+", r),
            Self::ReflexiveTransitiveClosure(r) => write!(f, "{}*", r),
            Self::Difference(a, b) => write!(f, "({} \\ {})", a, b),
            Self::DomainRestriction(r, d) => write!(f, "[{}]{}", d, r),
            Self::RangeRestriction(r, d) => write!(f, "{}[{}]", r, d),
            Self::Optional(r) => write!(f, "{}?", r),
            Self::Custom(expr) => write!(f, "{}", expr),
        }
    }
}

// ---------------------------------------------------------------------------
// Behavior & Behavior Sets
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Behavior {
    pub test_name: String,
    pub final_state: BTreeMap<String, u64>,
    pub read_values: BTreeMap<String, u64>,
}

impl Behavior {
    pub fn new(test_name: impl Into<String>) -> Self {
        Self {
            test_name: test_name.into(),
            final_state: BTreeMap::new(),
            read_values: BTreeMap::new(),
        }
    }

    pub fn with_final(mut self, addr: impl Into<String>, val: u64) -> Self {
        self.final_state.insert(addr.into(), val);
        self
    }

    pub fn with_read(mut self, reg: impl Into<String>, val: u64) -> Self {
        self.read_values.insert(reg.into(), val);
        self
    }

    pub fn fingerprint(&self) -> OutcomeFingerprint {
        let mut hasher = FnvHasher::new();
        hasher.feed_str(&self.test_name);
        for (k, v) in &self.final_state {
            hasher.feed_str(k);
            hasher.feed_u64(*v);
        }
        for (k, v) in &self.read_values {
            hasher.feed_str(k);
            hasher.feed_u64(*v);
        }
        OutcomeFingerprint(hasher.finish())
    }
}

impl fmt::Display for Behavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.final_state.iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .chain(self.read_values.iter().map(|(k, v)| format!("{}={}", k, v)))
            .collect();
        write!(f, "{{{}}}", parts.join(", "))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutcomeFingerprint(pub u64);

struct FnvHasher {
    state: u64,
}

impl FnvHasher {
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
pub struct BehaviorSet {
    pub model_name: String,
    pub behaviors: Vec<Behavior>,
    fingerprints: HashSet<OutcomeFingerprint>,
}

impl BehaviorSet {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            behaviors: Vec::new(),
            fingerprints: HashSet::new(),
        }
    }

    pub fn add(&mut self, behavior: Behavior) {
        let fp = behavior.fingerprint();
        if self.fingerprints.insert(fp) {
            self.behaviors.push(behavior);
        }
    }

    pub fn contains(&self, behavior: &Behavior) -> bool {
        self.fingerprints.contains(&behavior.fingerprint())
    }

    pub fn len(&self) -> usize {
        self.behaviors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.behaviors.is_empty()
    }

    pub fn union(&self, other: &BehaviorSet) -> BehaviorSet {
        let mut result = self.clone();
        for b in &other.behaviors {
            result.add(b.clone());
        }
        result
    }

    pub fn intersection(&self, other: &BehaviorSet) -> BehaviorSet {
        let mut result = BehaviorSet::new(format!("{} ∩ {}", self.model_name, other.model_name));
        for b in &self.behaviors {
            if other.contains(b) {
                result.add(b.clone());
            }
        }
        result
    }

    pub fn difference(&self, other: &BehaviorSet) -> BehaviorSet {
        let mut result = BehaviorSet::new(format!("{} \\ {}", self.model_name, other.model_name));
        for b in &self.behaviors {
            if !other.contains(b) {
                result.add(b.clone());
            }
        }
        result
    }

    pub fn symmetric_difference(&self, other: &BehaviorSet) -> BehaviorSet {
        let d1 = self.difference(other);
        let d2 = other.difference(self);
        d1.union(&d2)
    }

    pub fn is_subset_of(&self, other: &BehaviorSet) -> bool {
        self.behaviors.iter().all(|b| other.contains(b))
    }

    pub fn is_superset_of(&self, other: &BehaviorSet) -> bool {
        other.is_subset_of(self)
    }

    pub fn equals(&self, other: &BehaviorSet) -> bool {
        self.len() == other.len() && self.is_subset_of(other)
    }
}

impl fmt::Display for BehaviorSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BehaviorSet({}, {} behaviors)", self.model_name, self.behaviors.len())
    }
}

pub fn compute_behavior_set(
    model: &MemoryModelSpec,
    test_name: &str,
    outcomes: &[(BTreeMap<String, u64>, BTreeMap<String, u64>)],
) -> BehaviorSet {
    let mut set = BehaviorSet::new(&model.name);
    for (finals, reads) in outcomes {
        let mut b = Behavior::new(test_name);
        b.final_state = finals.clone();
        b.read_values = reads.clone();
        set.add(b);
    }
    set
}

// ---------------------------------------------------------------------------
// Model Diff
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffEntryKind {
    AxiomAdded,
    AxiomRemoved,
    AxiomModified,
    RelationAdded,
    RelationRemoved,
    RelationModified,
    ConstraintAdded,
    ConstraintRemoved,
    ConstraintModified,
    BehaviorAdded,
    BehaviorRemoved,
}

impl fmt::Display for DiffEntryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AxiomAdded => write!(f, "+axiom"),
            Self::AxiomRemoved => write!(f, "-axiom"),
            Self::AxiomModified => write!(f, "~axiom"),
            Self::RelationAdded => write!(f, "+relation"),
            Self::RelationRemoved => write!(f, "-relation"),
            Self::RelationModified => write!(f, "~relation"),
            Self::ConstraintAdded => write!(f, "+constraint"),
            Self::ConstraintRemoved => write!(f, "-constraint"),
            Self::ConstraintModified => write!(f, "~constraint"),
            Self::BehaviorAdded => write!(f, "+behavior"),
            Self::BehaviorRemoved => write!(f, "-behavior"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    pub kind: DiffEntryKind,
    pub name: String,
    pub description: String,
    pub details: Option<String>,
}

impl DiffEntry {
    pub fn new(kind: DiffEntryKind, name: impl Into<String>, desc: impl Into<String>) -> Self {
        Self { kind, name: name.into(), description: desc.into(), details: None }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    pub fn is_addition(&self) -> bool {
        matches!(self.kind,
            DiffEntryKind::AxiomAdded | DiffEntryKind::RelationAdded |
            DiffEntryKind::ConstraintAdded | DiffEntryKind::BehaviorAdded)
    }

    pub fn is_removal(&self) -> bool {
        matches!(self.kind,
            DiffEntryKind::AxiomRemoved | DiffEntryKind::RelationRemoved |
            DiffEntryKind::ConstraintRemoved | DiffEntryKind::BehaviorRemoved)
    }

    pub fn is_modification(&self) -> bool {
        matches!(self.kind,
            DiffEntryKind::AxiomModified | DiffEntryKind::RelationModified |
            DiffEntryKind::ConstraintModified)
    }
}

impl fmt::Display for DiffEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.kind, self.name, self.description)?;
        if let Some(ref details) = self.details {
            write!(f, " ({})", details)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDiff {
    pub model_a: String,
    pub model_b: String,
    pub entries: Vec<DiffEntry>,
}

impl ModelDiff {
    pub fn new(model_a: impl Into<String>, model_b: impl Into<String>) -> Self {
        Self { model_a: model_a.into(), model_b: model_b.into(), entries: Vec::new() }
    }

    pub fn add_entry(&mut self, entry: DiffEntry) {
        self.entries.push(entry);
    }

    pub fn additions(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.is_addition()).collect()
    }

    pub fn removals(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.is_removal()).collect()
    }

    pub fn modifications(&self) -> Vec<&DiffEntry> {
        self.entries.iter().filter(|e| e.is_modification()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn summary(&self) -> DiffSummary {
        DiffSummary {
            model_a: self.model_a.clone(),
            model_b: self.model_b.clone(),
            additions: self.additions().len(),
            removals: self.removals().len(),
            modifications: self.modifications().len(),
            total_changes: self.entries.len(),
        }
    }
}

impl fmt::Display for ModelDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Diff: {} vs {} ===", self.model_a, self.model_b)?;
        for entry in &self.entries {
            writeln!(f, "  {}", entry)?;
        }
        let summary = self.summary();
        writeln!(f, "--- {} additions, {} removals, {} modifications ---",
            summary.additions, summary.removals, summary.modifications)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub model_a: String,
    pub model_b: String,
    pub additions: usize,
    pub removals: usize,
    pub modifications: usize,
    pub total_changes: usize,
}

impl DiffSummary {
    pub fn is_identical(&self) -> bool {
        self.total_changes == 0
    }

    pub fn change_ratio(&self, total_elements: usize) -> f64 {
        if total_elements == 0 { return 0.0; }
        self.total_changes as f64 / total_elements as f64
    }
}

pub fn compute_diff(model_a: &MemoryModelSpec, model_b: &MemoryModelSpec) -> ModelDiff {
    let mut diff = ModelDiff::new(&model_a.name, &model_b.name);

    // Diff axioms
    let axioms_a: HashMap<&str, &AxiomSpec> = model_a.axioms.iter().map(|a| (a.name.as_str(), a)).collect();
    let axioms_b: HashMap<&str, &AxiomSpec> = model_b.axioms.iter().map(|a| (a.name.as_str(), a)).collect();

    for (name, spec_a) in &axioms_a {
        if let Some(spec_b) = axioms_b.get(name) {
            if spec_a != spec_b {
                diff.add_entry(DiffEntry::new(
                    DiffEntryKind::AxiomModified,
                    *name,
                    format!("Axiom '{}' modified", name),
                ).with_details(format!("{:?} -> {:?}", spec_a.kind, spec_b.kind)));
            }
        } else {
            diff.add_entry(DiffEntry::new(
                DiffEntryKind::AxiomRemoved,
                *name,
                format!("Axiom '{}' removed in {}", name, model_b.name),
            ));
        }
    }
    for name in axioms_b.keys() {
        if !axioms_a.contains_key(name) {
            diff.add_entry(DiffEntry::new(
                DiffEntryKind::AxiomAdded,
                *name,
                format!("Axiom '{}' added in {}", name, model_b.name),
            ));
        }
    }

    // Diff relations
    let rels_a: HashMap<&str, &RelationSpec> = model_a.relations.iter().map(|r| (r.name.as_str(), r)).collect();
    let rels_b: HashMap<&str, &RelationSpec> = model_b.relations.iter().map(|r| (r.name.as_str(), r)).collect();

    for (name, spec_a) in &rels_a {
        if let Some(spec_b) = rels_b.get(name) {
            if spec_a != spec_b {
                diff.add_entry(DiffEntry::new(
                    DiffEntryKind::RelationModified,
                    *name,
                    format!("Relation '{}' definition changed", name),
                ).with_details(format!("{} -> {}", spec_a.definition, spec_b.definition)));
            }
        } else {
            diff.add_entry(DiffEntry::new(
                DiffEntryKind::RelationRemoved,
                *name,
                format!("Relation '{}' removed in {}", name, model_b.name),
            ));
        }
    }
    for name in rels_b.keys() {
        if !rels_a.contains_key(name) {
            diff.add_entry(DiffEntry::new(
                DiffEntryKind::RelationAdded,
                *name,
                format!("Relation '{}' added in {}", name, model_b.name),
            ));
        }
    }

    // Diff constraints
    let cons_a: HashMap<&str, &ConstraintSpec> = model_a.constraints.iter().map(|c| (c.name.as_str(), c)).collect();
    let cons_b: HashMap<&str, &ConstraintSpec> = model_b.constraints.iter().map(|c| (c.name.as_str(), c)).collect();

    for (name, spec_a) in &cons_a {
        if let Some(spec_b) = cons_b.get(name) {
            if spec_a != spec_b {
                diff.add_entry(DiffEntry::new(
                    DiffEntryKind::ConstraintModified,
                    *name,
                    format!("Constraint '{}' modified", name),
                ));
            }
        } else {
            diff.add_entry(DiffEntry::new(
                DiffEntryKind::ConstraintRemoved,
                *name,
                format!("Constraint '{}' removed in {}", name, model_b.name),
            ));
        }
    }
    for name in cons_b.keys() {
        if !cons_a.contains_key(name) {
            diff.add_entry(DiffEntry::new(
                DiffEntryKind::ConstraintAdded,
                *name,
                format!("Constraint '{}' added in {}", name, model_b.name),
            ));
        }
    }

    diff
}

// ---------------------------------------------------------------------------
// Diff Matrix
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffMatrix {
    pub model_names: Vec<String>,
    pub diffs: Vec<Vec<DiffSummary>>,
}

impl DiffMatrix {
    pub fn size(&self) -> usize {
        self.model_names.len()
    }

    pub fn get(&self, i: usize, j: usize) -> &DiffSummary {
        &self.diffs[i][j]
    }

    pub fn most_similar_pair(&self) -> Option<(usize, usize)> {
        let n = self.model_names.len();
        let mut best = None;
        let mut best_changes = usize::MAX;
        for i in 0..n {
            for j in (i + 1)..n {
                let changes = self.diffs[i][j].total_changes;
                if changes < best_changes {
                    best_changes = changes;
                    best = Some((i, j));
                }
            }
        }
        best
    }

    pub fn most_different_pair(&self) -> Option<(usize, usize)> {
        let n = self.model_names.len();
        let mut best = None;
        let mut best_changes = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let changes = self.diffs[i][j].total_changes;
                if changes > best_changes {
                    best_changes = changes;
                    best = Some((i, j));
                }
            }
        }
        best
    }
}

pub fn diff_matrix(models: &[MemoryModelSpec]) -> DiffMatrix {
    let n = models.len();
    let names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    let mut diffs = Vec::with_capacity(n);

    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            if i == j {
                row.push(DiffSummary {
                    model_a: names[i].clone(),
                    model_b: names[j].clone(),
                    additions: 0,
                    removals: 0,
                    modifications: 0,
                    total_changes: 0,
                });
            } else {
                let d = compute_diff(&models[i], &models[j]);
                row.push(d.summary());
            }
        }
        diffs.push(row);
    }

    DiffMatrix { model_names: names, diffs }
}

// ---------------------------------------------------------------------------
// Subset Checking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsetResult {
    pub is_subset: bool,
    pub model_a: String,
    pub model_b: String,
    pub common_count: usize,
    pub extra_in_a: usize,
    pub extra_in_b: usize,
    pub counterexamples: Vec<SubsetCounterexample>,
}

impl SubsetResult {
    pub fn is_strict_subset(&self) -> bool {
        self.is_subset && self.extra_in_b > 0
    }
}

impl fmt::Display for SubsetResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_subset {
            write!(f, "{} ⊆ {} (common={}, extra_in_b={})",
                self.model_a, self.model_b, self.common_count, self.extra_in_b)
        } else {
            write!(f, "{} ⊄ {} ({} counterexamples)",
                self.model_a, self.model_b, self.counterexamples.len())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsetCounterexample {
    pub behavior: Behavior,
    pub description: String,
}

pub struct SubsetChecker;

impl SubsetChecker {
    pub fn is_behavioral_subset(set_a: &BehaviorSet, set_b: &BehaviorSet) -> SubsetResult {
        let common = set_a.intersection(set_b);
        let extra_a = set_a.difference(set_b);
        let extra_b = set_b.difference(set_a);

        let counterexamples: Vec<SubsetCounterexample> = extra_a.behaviors.iter()
            .map(|b| SubsetCounterexample {
                behavior: b.clone(),
                description: format!("Behavior {} is in {} but not in {}", b, set_a.model_name, set_b.model_name),
            })
            .collect();

        SubsetResult {
            is_subset: extra_a.is_empty(),
            model_a: set_a.model_name.clone(),
            model_b: set_b.model_name.clone(),
            common_count: common.len(),
            extra_in_a: extra_a.len(),
            extra_in_b: extra_b.len(),
            counterexamples,
        }
    }

    pub fn subset_matrix(sets: &[BehaviorSet]) -> Vec<Vec<bool>> {
        let n = sets.len();
        let mut matrix = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = sets[i].is_subset_of(&sets[j]);
            }
        }
        matrix
    }

    pub fn find_maximal_subsets(sets: &[BehaviorSet]) -> Vec<usize> {
        let n = sets.len();
        let matrix = Self::subset_matrix(sets);
        let mut maximal = Vec::new();

        for i in 0..n {
            let mut is_maximal = true;
            for j in 0..n {
                if i != j && matrix[i][j] && !matrix[j][i] {
                    // i is strict subset of j => not maximal
                    is_maximal = false;
                    break;
                }
            }
            if is_maximal {
                maximal.push(i);
            }
        }
        maximal
    }
}

// ---------------------------------------------------------------------------
// Refinement Checking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementResult {
    pub is_refinement: bool,
    pub spec_name: String,
    pub impl_name: String,
    pub violations: Vec<RefinementViolation>,
    pub coverage: f64,
}

impl RefinementResult {
    pub fn is_sound(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn is_complete(&self) -> bool {
        self.coverage >= 1.0 - 1e-10
    }
}

impl fmt::Display for RefinementResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_refinement {
            write!(f, "{} refines {} (coverage={:.1}%)",
                self.impl_name, self.spec_name, self.coverage * 100.0)
        } else {
            write!(f, "{} does NOT refine {} ({} violations)",
                self.impl_name, self.spec_name, self.violations.len())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementViolation {
    pub behavior: Behavior,
    pub violation_kind: RefinementViolationKind,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefinementViolationKind {
    ExtraBehavior,
    MissingBehavior,
    AxiomViolation,
}

impl fmt::Display for RefinementViolationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExtraBehavior => write!(f, "extra_behavior"),
            Self::MissingBehavior => write!(f, "missing_behavior"),
            Self::AxiomViolation => write!(f, "axiom_violation"),
        }
    }
}

pub struct RefinementChecker;

impl RefinementChecker {
    pub fn is_refinement(
        spec_behaviors: &BehaviorSet,
        impl_behaviors: &BehaviorSet,
    ) -> RefinementResult {
        let extra = impl_behaviors.difference(spec_behaviors);
        let missing = spec_behaviors.difference(impl_behaviors);
        let common = spec_behaviors.intersection(impl_behaviors);

        let violations: Vec<RefinementViolation> = extra.behaviors.iter()
            .map(|b| RefinementViolation {
                behavior: b.clone(),
                violation_kind: RefinementViolationKind::ExtraBehavior,
                description: format!(
                    "Implementation allows behavior {} not in spec",
                    b
                ),
            })
            .collect();

        let coverage = if spec_behaviors.len() > 0 {
            common.len() as f64 / spec_behaviors.len() as f64
        } else {
            1.0
        };

        RefinementResult {
            is_refinement: extra.is_empty(),
            spec_name: spec_behaviors.model_name.clone(),
            impl_name: impl_behaviors.model_name.clone(),
            violations,
            coverage,
        }
    }

    pub fn refinement_chain(sets: &[BehaviorSet]) -> Vec<(usize, usize)> {
        let n = sets.len();
        let mut chain = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i != j && sets[i].is_subset_of(&sets[j]) {
                    chain.push((i, j));
                }
            }
        }
        chain
    }
}

// ---------------------------------------------------------------------------
// Diff Formatter
// ---------------------------------------------------------------------------

pub struct DiffFormatter;

impl DiffFormatter {
    pub fn format_diff_table(diff: &ModelDiff) -> String {
        let mut output = String::new();
        output.push_str(&format!("┌─── {} vs {} ───┐\n", diff.model_a, diff.model_b));
        output.push_str("│ Type       │ Name              │ Detail                         │\n");
        output.push_str("├────────────┼───────────────────┼────────────────────────────────┤\n");

        for entry in &diff.entries {
            let kind_str = format!("{}", entry.kind);
            let detail = entry.details.as_deref().unwrap_or("");
            output.push_str(&format!("│ {:10} │ {:17} │ {:30} │\n",
                &kind_str[..kind_str.len().min(10)],
                &entry.name[..entry.name.len().min(17)],
                &detail[..detail.len().min(30)],
            ));
        }

        output.push_str("└────────────┴───────────────────┴────────────────────────────────┘\n");
        let summary = diff.summary();
        output.push_str(&format!("Total: +{} -{} ~{}\n",
            summary.additions, summary.removals, summary.modifications));
        output
    }

    pub fn format_venn_diagram(
        set_a: &BehaviorSet,
        set_b: &BehaviorSet,
    ) -> String {
        let only_a = set_a.difference(set_b).len();
        let only_b = set_b.difference(set_a).len();
        let common = set_a.intersection(set_b).len();

        let mut output = String::new();
        output.push_str(&format!("    {} only: {}\n", set_a.model_name, only_a));
        output.push_str(&format!("    Common:    {}\n", common));
        output.push_str(&format!("    {} only: {}\n", set_b.model_name, only_b));
        output.push_str("\n");

        // ASCII art Venn diagram
        let total = only_a + only_b + common;
        if total > 0 {
            let a_pct = (only_a as f64 / total as f64 * 20.0) as usize;
            let c_pct = (common as f64 / total as f64 * 20.0) as usize;
            let b_pct = (only_b as f64 / total as f64 * 20.0) as usize;

            output.push_str("    ");
            output.push_str(&"A".repeat(a_pct.max(1)));
            output.push_str(&"*".repeat(c_pct.max(1)));
            output.push_str(&"B".repeat(b_pct.max(1)));
            output.push('\n');
        }

        output
    }

    pub fn format_summary_table(matrix: &DiffMatrix) -> String {
        let mut output = String::new();
        let n = matrix.size();

        // Header
        output.push_str(&format!("{:15}", ""));
        for name in &matrix.model_names {
            output.push_str(&format!("{:10}", &name[..name.len().min(10)]));
        }
        output.push('\n');

        for i in 0..n {
            output.push_str(&format!("{:15}", &matrix.model_names[i][..matrix.model_names[i].len().min(15)]));
            for j in 0..n {
                let changes = matrix.diffs[i][j].total_changes;
                output.push_str(&format!("{:10}", changes));
            }
            output.push('\n');
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Discriminators
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Discriminator {
    pub test_name: String,
    pub behavior: Behavior,
    pub allowed_by: Vec<String>,
    pub forbidden_by: Vec<String>,
}

impl Discriminator {
    pub fn discriminates(&self, model_a: &str, model_b: &str) -> bool {
        (self.allowed_by.contains(&model_a.to_string()) && self.forbidden_by.contains(&model_b.to_string())) ||
        (self.allowed_by.contains(&model_b.to_string()) && self.forbidden_by.contains(&model_a.to_string()))
    }

    pub fn strength(&self) -> usize {
        self.allowed_by.len().min(self.forbidden_by.len())
    }
}

impl fmt::Display for Discriminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Discriminator({}: allowed by {:?}, forbidden by {:?})",
            self.test_name, self.allowed_by, self.forbidden_by)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscriminatorSet {
    pub discriminators: Vec<Discriminator>,
}

impl DiscriminatorSet {
    pub fn new() -> Self {
        Self { discriminators: Vec::new() }
    }

    pub fn add(&mut self, disc: Discriminator) {
        self.discriminators.push(disc);
    }

    pub fn len(&self) -> usize {
        self.discriminators.len()
    }

    pub fn is_empty(&self) -> bool {
        self.discriminators.is_empty()
    }

    pub fn for_pair(&self, model_a: &str, model_b: &str) -> Vec<&Discriminator> {
        self.discriminators.iter()
            .filter(|d| d.discriminates(model_a, model_b))
            .collect()
    }

    pub fn strongest(&self) -> Option<&Discriminator> {
        self.discriminators.iter().max_by_key(|d| d.strength())
    }
}

pub fn find_discriminators(
    model_behaviors: &[(&str, &BehaviorSet)],
) -> DiscriminatorSet {
    let mut disc_set = DiscriminatorSet::new();

    // Collect all unique behaviors across all models
    let mut all_behaviors: HashSet<OutcomeFingerprint> = HashSet::new();
    let mut behavior_map: HashMap<OutcomeFingerprint, Behavior> = HashMap::new();

    for (_, set) in model_behaviors {
        for b in &set.behaviors {
            let fp = b.fingerprint();
            all_behaviors.insert(fp);
            behavior_map.entry(fp).or_insert_with(|| b.clone());
        }
    }

    // For each behavior, determine which models allow/forbid it
    for (fp, behavior) in &behavior_map {
        let mut allowed_by = Vec::new();
        let mut forbidden_by = Vec::new();

        for (name, set) in model_behaviors {
            if set.contains(behavior) {
                allowed_by.push(name.to_string());
            } else {
                forbidden_by.push(name.to_string());
            }
        }

        // Only a discriminator if some models allow it and some forbid it
        if !allowed_by.is_empty() && !forbidden_by.is_empty() {
            disc_set.add(Discriminator {
                test_name: behavior.test_name.clone(),
                behavior: behavior.clone(),
                allowed_by,
                forbidden_by,
            });
        }
    }

    disc_set
}

pub fn minimal_discriminator(
    model_a: &str,
    model_b: &str,
    disc_set: &DiscriminatorSet,
) -> Option<Discriminator> {
    let candidates = disc_set.for_pair(model_a, model_b);
    candidates.into_iter().min_by_key(|d| {
        // Prefer discriminators with simpler behaviors
        d.behavior.final_state.len() + d.behavior.read_values.len()
    }).cloned()
}

// ---------------------------------------------------------------------------
// Model Distance
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceMatrix {
    pub model_names: Vec<String>,
    pub distances: Vec<Vec<f64>>,
}

impl DistanceMatrix {
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.distances[i][j]
    }

    pub fn closest_pair(&self) -> Option<(usize, usize, f64)> {
        let n = self.model_names.len();
        let mut best = None;
        let mut best_dist = f64::INFINITY;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.distances[i][j] < best_dist {
                    best_dist = self.distances[i][j];
                    best = Some((i, j, best_dist));
                }
            }
        }
        best
    }

    pub fn farthest_pair(&self) -> Option<(usize, usize, f64)> {
        let n = self.model_names.len();
        let mut best = None;
        let mut best_dist = f64::NEG_INFINITY;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.distances[i][j] > best_dist {
                    best_dist = self.distances[i][j];
                    best = Some((i, j, best_dist));
                }
            }
        }
        best
    }

    pub fn average_distance(&self) -> f64 {
        let n = self.model_names.len();
        if n < 2 { return 0.0; }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += self.distances[i][j];
                count += 1;
            }
        }
        total / count as f64
    }

    pub fn hierarchical_clustering(&self) -> Vec<(usize, usize, f64)> {
        let n = self.model_names.len();
        let mut active: Vec<bool> = vec![true; n];
        let mut distances = self.distances.clone();
        let mut merges = Vec::new();

        for _ in 0..n.saturating_sub(1) {
            // Find closest active pair
            let mut best_dist = f64::INFINITY;
            let mut best_i = 0;
            let mut best_j = 0;
            for i in 0..n {
                if !active[i] { continue; }
                for j in (i + 1)..n {
                    if !active[j] { continue; }
                    if distances[i][j] < best_dist {
                        best_dist = distances[i][j];
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if best_dist == f64::INFINITY { break; }

            merges.push((best_i, best_j, best_dist));
            active[best_j] = false;

            // Update distances (single-linkage)
            for k in 0..n {
                if !active[k] || k == best_i { continue; }
                distances[best_i][k] = distances[best_i][k].min(distances[best_j][k]);
                distances[k][best_i] = distances[best_i][k];
            }
        }

        merges
    }
}

pub struct ModelDistance;

impl ModelDistance {
    pub fn jaccard_distance(set_a: &BehaviorSet, set_b: &BehaviorSet) -> f64 {
        let common = set_a.intersection(set_b).len();
        let union_size = set_a.union(set_b).len();
        if union_size == 0 { return 0.0; }
        1.0 - (common as f64 / union_size as f64)
    }

    pub fn hamming_distance(set_a: &BehaviorSet, set_b: &BehaviorSet) -> f64 {
        let only_a = set_a.difference(set_b).len();
        let only_b = set_b.difference(set_a).len();
        (only_a + only_b) as f64
    }

    pub fn dice_distance(set_a: &BehaviorSet, set_b: &BehaviorSet) -> f64 {
        let common = set_a.intersection(set_b).len();
        let total = set_a.len() + set_b.len();
        if total == 0 { return 0.0; }
        1.0 - (2.0 * common as f64 / total as f64)
    }

    pub fn overlap_coefficient(set_a: &BehaviorSet, set_b: &BehaviorSet) -> f64 {
        let common = set_a.intersection(set_b).len();
        let min_size = set_a.len().min(set_b.len());
        if min_size == 0 { return 0.0; }
        common as f64 / min_size as f64
    }

    pub fn structural_distance(model_a: &MemoryModelSpec, model_b: &MemoryModelSpec) -> f64 {
        let diff = compute_diff(model_a, model_b);
        let total = model_a.axioms.len() + model_a.relations.len() + model_a.constraints.len()
            + model_b.axioms.len() + model_b.relations.len() + model_b.constraints.len();
        if total == 0 { return 0.0; }
        diff.entries.len() as f64 / (total as f64 / 2.0)
    }
}

pub fn distance_matrix(
    sets: &[(&str, &BehaviorSet)],
    metric: fn(&BehaviorSet, &BehaviorSet) -> f64,
) -> DistanceMatrix {
    let n = sets.len();
    let names: Vec<String> = sets.iter().map(|(name, _)| name.to_string()).collect();
    let mut distances = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = metric(sets[i].1, sets[j].1);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }

    DistanceMatrix { model_names: names, distances }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// ===== Extended Diff Operations =====

#[derive(Debug, Clone)]
pub struct DiffMerge {
    pub left_changes: Vec<String>,
    pub right_changes: Vec<String>,
    pub conflicts: Vec<String>,
}

impl DiffMerge {
    pub fn new(left_changes: Vec<String>, right_changes: Vec<String>, conflicts: Vec<String>) -> Self {
        DiffMerge { left_changes, right_changes, conflicts }
    }

    pub fn left_changes_len(&self) -> usize {
        self.left_changes.len()
    }

    pub fn left_changes_is_empty(&self) -> bool {
        self.left_changes.is_empty()
    }

    pub fn right_changes_len(&self) -> usize {
        self.right_changes.len()
    }

    pub fn right_changes_is_empty(&self) -> bool {
        self.right_changes.is_empty()
    }

    pub fn conflicts_len(&self) -> usize {
        self.conflicts.len()
    }

    pub fn conflicts_is_empty(&self) -> bool {
        self.conflicts.is_empty()
    }

}

impl fmt::Display for DiffMerge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffMerge({:?})", self.left_changes)
    }
}

#[derive(Debug, Clone)]
pub struct DiffMergeBuilder {
    left_changes: Vec<String>,
    right_changes: Vec<String>,
    conflicts: Vec<String>,
}

impl DiffMergeBuilder {
    pub fn new() -> Self {
        DiffMergeBuilder {
            left_changes: Vec::new(),
            right_changes: Vec::new(),
            conflicts: Vec::new(),
        }
    }

    pub fn left_changes(mut self, v: Vec<String>) -> Self { self.left_changes = v; self }
    pub fn right_changes(mut self, v: Vec<String>) -> Self { self.right_changes = v; self }
    pub fn conflicts(mut self, v: Vec<String>) -> Self { self.conflicts = v; self }
}

#[derive(Debug, Clone)]
pub struct ThreeWayDiff {
    pub base_axioms: Vec<String>,
    pub left_axioms: Vec<String>,
    pub right_axioms: Vec<String>,
    pub merged: Vec<String>,
}

impl ThreeWayDiff {
    pub fn new(base_axioms: Vec<String>, left_axioms: Vec<String>, right_axioms: Vec<String>, merged: Vec<String>) -> Self {
        ThreeWayDiff { base_axioms, left_axioms, right_axioms, merged }
    }

    pub fn base_axioms_len(&self) -> usize {
        self.base_axioms.len()
    }

    pub fn base_axioms_is_empty(&self) -> bool {
        self.base_axioms.is_empty()
    }

    pub fn left_axioms_len(&self) -> usize {
        self.left_axioms.len()
    }

    pub fn left_axioms_is_empty(&self) -> bool {
        self.left_axioms.is_empty()
    }

    pub fn right_axioms_len(&self) -> usize {
        self.right_axioms.len()
    }

    pub fn right_axioms_is_empty(&self) -> bool {
        self.right_axioms.is_empty()
    }

    pub fn merged_len(&self) -> usize {
        self.merged.len()
    }

    pub fn merged_is_empty(&self) -> bool {
        self.merged.is_empty()
    }

}

impl fmt::Display for ThreeWayDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ThreeWayDiff({:?})", self.base_axioms)
    }
}

#[derive(Debug, Clone)]
pub struct ThreeWayDiffBuilder {
    base_axioms: Vec<String>,
    left_axioms: Vec<String>,
    right_axioms: Vec<String>,
    merged: Vec<String>,
}

impl ThreeWayDiffBuilder {
    pub fn new() -> Self {
        ThreeWayDiffBuilder {
            base_axioms: Vec::new(),
            left_axioms: Vec::new(),
            right_axioms: Vec::new(),
            merged: Vec::new(),
        }
    }

    pub fn base_axioms(mut self, v: Vec<String>) -> Self { self.base_axioms = v; self }
    pub fn left_axioms(mut self, v: Vec<String>) -> Self { self.left_axioms = v; self }
    pub fn right_axioms(mut self, v: Vec<String>) -> Self { self.right_axioms = v; self }
    pub fn merged(mut self, v: Vec<String>) -> Self { self.merged = v; self }
}

#[derive(Debug, Clone)]
pub struct DiffCompression {
    pub original_size: usize,
    pub compressed_size: usize,
    pub ratio: f64,
}

impl DiffCompression {
    pub fn new(original_size: usize, compressed_size: usize, ratio: f64) -> Self {
        DiffCompression { original_size, compressed_size, ratio }
    }

    pub fn get_original_size(&self) -> usize {
        self.original_size
    }

    pub fn get_compressed_size(&self) -> usize {
        self.compressed_size
    }

    pub fn get_ratio(&self) -> f64 {
        self.ratio
    }

    pub fn with_original_size(mut self, v: usize) -> Self {
        self.original_size = v; self
    }

    pub fn with_compressed_size(mut self, v: usize) -> Self {
        self.compressed_size = v; self
    }

    pub fn with_ratio(mut self, v: f64) -> Self {
        self.ratio = v; self
    }

}

impl fmt::Display for DiffCompression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffCompression({:?})", self.original_size)
    }
}

#[derive(Debug, Clone)]
pub struct DiffCompressionBuilder {
    original_size: usize,
    compressed_size: usize,
    ratio: f64,
}

impl DiffCompressionBuilder {
    pub fn new() -> Self {
        DiffCompressionBuilder {
            original_size: 0,
            compressed_size: 0,
            ratio: 0.0,
        }
    }

    pub fn original_size(mut self, v: usize) -> Self { self.original_size = v; self }
    pub fn compressed_size(mut self, v: usize) -> Self { self.compressed_size = v; self }
    pub fn ratio(mut self, v: f64) -> Self { self.ratio = v; self }
}

#[derive(Debug, Clone)]
pub struct ModelComposition {
    pub model_a: String,
    pub model_b: String,
    pub composed: String,
    pub axiom_count: usize,
}

impl ModelComposition {
    pub fn new(model_a: String, model_b: String, composed: String, axiom_count: usize) -> Self {
        ModelComposition { model_a, model_b, composed, axiom_count }
    }

    pub fn get_model_a(&self) -> &str {
        &self.model_a
    }

    pub fn get_model_b(&self) -> &str {
        &self.model_b
    }

    pub fn get_composed(&self) -> &str {
        &self.composed
    }

    pub fn get_axiom_count(&self) -> usize {
        self.axiom_count
    }

    pub fn with_model_a(mut self, v: impl Into<String>) -> Self {
        self.model_a = v.into(); self
    }

    pub fn with_model_b(mut self, v: impl Into<String>) -> Self {
        self.model_b = v.into(); self
    }

    pub fn with_composed(mut self, v: impl Into<String>) -> Self {
        self.composed = v.into(); self
    }

    pub fn with_axiom_count(mut self, v: usize) -> Self {
        self.axiom_count = v; self
    }

}

impl fmt::Display for ModelComposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelComposition({:?})", self.model_a)
    }
}

#[derive(Debug, Clone)]
pub struct ModelCompositionBuilder {
    model_a: String,
    model_b: String,
    composed: String,
    axiom_count: usize,
}

impl ModelCompositionBuilder {
    pub fn new() -> Self {
        ModelCompositionBuilder {
            model_a: String::new(),
            model_b: String::new(),
            composed: String::new(),
            axiom_count: 0,
        }
    }

    pub fn model_a(mut self, v: impl Into<String>) -> Self { self.model_a = v.into(); self }
    pub fn model_b(mut self, v: impl Into<String>) -> Self { self.model_b = v.into(); self }
    pub fn composed(mut self, v: impl Into<String>) -> Self { self.composed = v.into(); self }
    pub fn axiom_count(mut self, v: usize) -> Self { self.axiom_count = v; self }
}

#[derive(Debug, Clone)]
pub struct ModelWeakening {
    pub removed_axioms: Vec<String>,
    pub weakened_constraints: Vec<String>,
    pub strength_delta: f64,
}

impl ModelWeakening {
    pub fn new(removed_axioms: Vec<String>, weakened_constraints: Vec<String>, strength_delta: f64) -> Self {
        ModelWeakening { removed_axioms, weakened_constraints, strength_delta }
    }

    pub fn removed_axioms_len(&self) -> usize {
        self.removed_axioms.len()
    }

    pub fn removed_axioms_is_empty(&self) -> bool {
        self.removed_axioms.is_empty()
    }

    pub fn weakened_constraints_len(&self) -> usize {
        self.weakened_constraints.len()
    }

    pub fn weakened_constraints_is_empty(&self) -> bool {
        self.weakened_constraints.is_empty()
    }

    pub fn get_strength_delta(&self) -> f64 {
        self.strength_delta
    }

    pub fn with_strength_delta(mut self, v: f64) -> Self {
        self.strength_delta = v; self
    }

}

impl fmt::Display for ModelWeakening {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelWeakening({:?})", self.removed_axioms)
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeakeningBuilder {
    removed_axioms: Vec<String>,
    weakened_constraints: Vec<String>,
    strength_delta: f64,
}

impl ModelWeakeningBuilder {
    pub fn new() -> Self {
        ModelWeakeningBuilder {
            removed_axioms: Vec::new(),
            weakened_constraints: Vec::new(),
            strength_delta: 0.0,
        }
    }

    pub fn removed_axioms(mut self, v: Vec<String>) -> Self { self.removed_axioms = v; self }
    pub fn weakened_constraints(mut self, v: Vec<String>) -> Self { self.weakened_constraints = v; self }
    pub fn strength_delta(mut self, v: f64) -> Self { self.strength_delta = v; self }
}

#[derive(Debug, Clone)]
pub struct ModelStrengthening {
    pub added_axioms: Vec<String>,
    pub strengthened_constraints: Vec<String>,
    pub strength_delta: f64,
}

impl ModelStrengthening {
    pub fn new(added_axioms: Vec<String>, strengthened_constraints: Vec<String>, strength_delta: f64) -> Self {
        ModelStrengthening { added_axioms, strengthened_constraints, strength_delta }
    }

    pub fn added_axioms_len(&self) -> usize {
        self.added_axioms.len()
    }

    pub fn added_axioms_is_empty(&self) -> bool {
        self.added_axioms.is_empty()
    }

    pub fn strengthened_constraints_len(&self) -> usize {
        self.strengthened_constraints.len()
    }

    pub fn strengthened_constraints_is_empty(&self) -> bool {
        self.strengthened_constraints.is_empty()
    }

    pub fn get_strength_delta(&self) -> f64 {
        self.strength_delta
    }

    pub fn with_strength_delta(mut self, v: f64) -> Self {
        self.strength_delta = v; self
    }

}

impl fmt::Display for ModelStrengthening {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelStrengthening({:?})", self.added_axioms)
    }
}

#[derive(Debug, Clone)]
pub struct ModelStrengtheningBuilder {
    added_axioms: Vec<String>,
    strengthened_constraints: Vec<String>,
    strength_delta: f64,
}

impl ModelStrengtheningBuilder {
    pub fn new() -> Self {
        ModelStrengtheningBuilder {
            added_axioms: Vec::new(),
            strengthened_constraints: Vec::new(),
            strength_delta: 0.0,
        }
    }

    pub fn added_axioms(mut self, v: Vec<String>) -> Self { self.added_axioms = v; self }
    pub fn strengthened_constraints(mut self, v: Vec<String>) -> Self { self.strengthened_constraints = v; self }
    pub fn strength_delta(mut self, v: f64) -> Self { self.strength_delta = v; self }
}

#[derive(Debug, Clone)]
pub struct LitmusTransformation {
    pub original_test: String,
    pub transformed_test: String,
    pub transform_type: String,
}

impl LitmusTransformation {
    pub fn new(original_test: String, transformed_test: String, transform_type: String) -> Self {
        LitmusTransformation { original_test, transformed_test, transform_type }
    }

    pub fn get_original_test(&self) -> &str {
        &self.original_test
    }

    pub fn get_transformed_test(&self) -> &str {
        &self.transformed_test
    }

    pub fn get_transform_type(&self) -> &str {
        &self.transform_type
    }

    pub fn with_original_test(mut self, v: impl Into<String>) -> Self {
        self.original_test = v.into(); self
    }

    pub fn with_transformed_test(mut self, v: impl Into<String>) -> Self {
        self.transformed_test = v.into(); self
    }

    pub fn with_transform_type(mut self, v: impl Into<String>) -> Self {
        self.transform_type = v.into(); self
    }

}

impl fmt::Display for LitmusTransformation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LitmusTransformation({:?})", self.original_test)
    }
}

#[derive(Debug, Clone)]
pub struct LitmusTransformationBuilder {
    original_test: String,
    transformed_test: String,
    transform_type: String,
}

impl LitmusTransformationBuilder {
    pub fn new() -> Self {
        LitmusTransformationBuilder {
            original_test: String::new(),
            transformed_test: String::new(),
            transform_type: String::new(),
        }
    }

    pub fn original_test(mut self, v: impl Into<String>) -> Self { self.original_test = v.into(); self }
    pub fn transformed_test(mut self, v: impl Into<String>) -> Self { self.transformed_test = v.into(); self }
    pub fn transform_type(mut self, v: impl Into<String>) -> Self { self.transform_type = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct OutcomePrediction {
    pub model_name: String,
    pub predicted_outcomes: Vec<String>,
    pub confidence: f64,
}

impl OutcomePrediction {
    pub fn new(model_name: String, predicted_outcomes: Vec<String>, confidence: f64) -> Self {
        OutcomePrediction { model_name, predicted_outcomes, confidence }
    }

    pub fn get_model_name(&self) -> &str {
        &self.model_name
    }

    pub fn predicted_outcomes_len(&self) -> usize {
        self.predicted_outcomes.len()
    }

    pub fn predicted_outcomes_is_empty(&self) -> bool {
        self.predicted_outcomes.is_empty()
    }

    pub fn get_confidence(&self) -> f64 {
        self.confidence
    }

    pub fn with_model_name(mut self, v: impl Into<String>) -> Self {
        self.model_name = v.into(); self
    }

    pub fn with_confidence(mut self, v: f64) -> Self {
        self.confidence = v; self
    }

}

impl fmt::Display for OutcomePrediction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OutcomePrediction({:?})", self.model_name)
    }
}

#[derive(Debug, Clone)]
pub struct OutcomePredictionBuilder {
    model_name: String,
    predicted_outcomes: Vec<String>,
    confidence: f64,
}

impl OutcomePredictionBuilder {
    pub fn new() -> Self {
        OutcomePredictionBuilder {
            model_name: String::new(),
            predicted_outcomes: Vec::new(),
            confidence: 0.0,
        }
    }

    pub fn model_name(mut self, v: impl Into<String>) -> Self { self.model_name = v.into(); self }
    pub fn predicted_outcomes(mut self, v: Vec<String>) -> Self { self.predicted_outcomes = v; self }
    pub fn confidence(mut self, v: f64) -> Self { self.confidence = v; self }
}

#[derive(Debug, Clone)]
pub struct BehaviorCounter {
    pub total_behaviors: u64,
    pub allowed_behaviors: u64,
    pub forbidden_behaviors: u64,
}

impl BehaviorCounter {
    pub fn new(total_behaviors: u64, allowed_behaviors: u64, forbidden_behaviors: u64) -> Self {
        BehaviorCounter { total_behaviors, allowed_behaviors, forbidden_behaviors }
    }

    pub fn get_total_behaviors(&self) -> u64 {
        self.total_behaviors
    }

    pub fn get_allowed_behaviors(&self) -> u64 {
        self.allowed_behaviors
    }

    pub fn get_forbidden_behaviors(&self) -> u64 {
        self.forbidden_behaviors
    }

    pub fn with_total_behaviors(mut self, v: u64) -> Self {
        self.total_behaviors = v; self
    }

    pub fn with_allowed_behaviors(mut self, v: u64) -> Self {
        self.allowed_behaviors = v; self
    }

    pub fn with_forbidden_behaviors(mut self, v: u64) -> Self {
        self.forbidden_behaviors = v; self
    }

}

impl fmt::Display for BehaviorCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BehaviorCounter({:?})", self.total_behaviors)
    }
}

#[derive(Debug, Clone)]
pub struct BehaviorCounterBuilder {
    total_behaviors: u64,
    allowed_behaviors: u64,
    forbidden_behaviors: u64,
}

impl BehaviorCounterBuilder {
    pub fn new() -> Self {
        BehaviorCounterBuilder {
            total_behaviors: 0,
            allowed_behaviors: 0,
            forbidden_behaviors: 0,
        }
    }

    pub fn total_behaviors(mut self, v: u64) -> Self { self.total_behaviors = v; self }
    pub fn allowed_behaviors(mut self, v: u64) -> Self { self.allowed_behaviors = v; self }
    pub fn forbidden_behaviors(mut self, v: u64) -> Self { self.forbidden_behaviors = v; self }
}

#[derive(Debug, Clone)]
pub struct ModelFeatureVector {
    pub features: Vec<f64>,
    pub dimension: usize,
    pub label: String,
}

impl ModelFeatureVector {
    pub fn new(features: Vec<f64>, dimension: usize, label: String) -> Self {
        ModelFeatureVector { features, dimension, label }
    }

    pub fn features_len(&self) -> usize {
        self.features.len()
    }

    pub fn features_is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    pub fn get_label(&self) -> &str {
        &self.label
    }

    pub fn with_dimension(mut self, v: usize) -> Self {
        self.dimension = v; self
    }

    pub fn with_label(mut self, v: impl Into<String>) -> Self {
        self.label = v.into(); self
    }

}

impl fmt::Display for ModelFeatureVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelFeatureVector({:?})", self.features)
    }
}

#[derive(Debug, Clone)]
pub struct ModelFeatureVectorBuilder {
    features: Vec<f64>,
    dimension: usize,
    label: String,
}

impl ModelFeatureVectorBuilder {
    pub fn new() -> Self {
        ModelFeatureVectorBuilder {
            features: Vec::new(),
            dimension: 0,
            label: String::new(),
        }
    }

    pub fn features(mut self, v: Vec<f64>) -> Self { self.features = v; self }
    pub fn dimension(mut self, v: usize) -> Self { self.dimension = v; self }
    pub fn label(mut self, v: impl Into<String>) -> Self { self.label = v.into(); self }
}

#[derive(Debug, Clone)]
pub struct DiffPatch {
    pub operations: Vec<String>,
    pub applied: bool,
    pub reversible: bool,
}

impl DiffPatch {
    pub fn new(operations: Vec<String>, applied: bool, reversible: bool) -> Self {
        DiffPatch { operations, applied, reversible }
    }

    pub fn operations_len(&self) -> usize {
        self.operations.len()
    }

    pub fn operations_is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    pub fn get_applied(&self) -> bool {
        self.applied
    }

    pub fn get_reversible(&self) -> bool {
        self.reversible
    }

    pub fn with_applied(mut self, v: bool) -> Self {
        self.applied = v; self
    }

    pub fn with_reversible(mut self, v: bool) -> Self {
        self.reversible = v; self
    }

}

impl fmt::Display for DiffPatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffPatch({:?})", self.operations)
    }
}

#[derive(Debug, Clone)]
pub struct DiffPatchBuilder {
    operations: Vec<String>,
    applied: bool,
    reversible: bool,
}

impl DiffPatchBuilder {
    pub fn new() -> Self {
        DiffPatchBuilder {
            operations: Vec::new(),
            applied: false,
            reversible: false,
        }
    }

    pub fn operations(mut self, v: Vec<String>) -> Self { self.operations = v; self }
    pub fn applied(mut self, v: bool) -> Self { self.applied = v; self }
    pub fn reversible(mut self, v: bool) -> Self { self.reversible = v; self }
}

#[derive(Debug, Clone)]
pub struct ModelDistanceExt {
    pub metric: String,
    pub distance: f64,
    pub normalized: bool,
}

impl ModelDistanceExt {
    pub fn new(metric: String, distance: f64, normalized: bool) -> Self {
        ModelDistanceExt { metric, distance, normalized }
    }

    pub fn get_metric(&self) -> &str {
        &self.metric
    }

    pub fn get_distance(&self) -> f64 {
        self.distance
    }

    pub fn get_normalized(&self) -> bool {
        self.normalized
    }

    pub fn with_metric(mut self, v: impl Into<String>) -> Self {
        self.metric = v.into(); self
    }

    pub fn with_distance(mut self, v: f64) -> Self {
        self.distance = v; self
    }

    pub fn with_normalized(mut self, v: bool) -> Self {
        self.normalized = v; self
    }

    pub fn jaccard_distance(a: &BehaviorSet, b: &BehaviorSet) -> f64 {
        let union_size = a.union(b).len();
        if union_size == 0 { return 0.0; }
        let inter_size = a.intersection(b).len();
        1.0 - (inter_size as f64 / union_size as f64)
    }

    pub fn hamming_distance(a: &BehaviorSet, b: &BehaviorSet) -> f64 {
        a.symmetric_difference(b).len() as f64
    }

}

impl fmt::Display for ModelDistanceExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModelDistanceExt({:?})", self.metric)
    }
}

#[derive(Debug, Clone)]
pub struct ModelDistanceExtBuilder {
    metric: String,
    distance: f64,
    normalized: bool,
}

impl ModelDistanceExtBuilder {
    pub fn new() -> Self {
        ModelDistanceExtBuilder {
            metric: String::new(),
            distance: 0.0,
            normalized: false,
        }
    }

    pub fn metric(mut self, v: impl Into<String>) -> Self { self.metric = v.into(); self }
    pub fn distance(mut self, v: f64) -> Self { self.distance = v; self }
    pub fn normalized(mut self, v: bool) -> Self { self.normalized = v; self }
}

#[derive(Debug, Clone)]
pub struct DiffAnalysis {
    pub data: Vec<Vec<f64>>,
    pub size: usize,
    pub computed: bool,
    pub label: String,
    pub threshold: f64,
}

impl DiffAnalysis {
    pub fn new(size: usize) -> Self {
        let data = vec![vec![0.0; size]; size];
        DiffAnalysis { data, size, computed: false, label: "Diff".to_string(), threshold: 0.01 }
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

impl fmt::Display for DiffAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffAnalysis({:?})", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiffStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Skipped,
}

impl fmt::Display for DiffStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffStatus::Pending => write!(f, "pending"),
            DiffStatus::InProgress => write!(f, "inprogress"),
            DiffStatus::Completed => write!(f, "completed"),
            DiffStatus::Failed => write!(f, "failed"),
            DiffStatus::Skipped => write!(f, "skipped"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiffPriority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl fmt::Display for DiffPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffPriority::Critical => write!(f, "critical"),
            DiffPriority::High => write!(f, "high"),
            DiffPriority::Medium => write!(f, "medium"),
            DiffPriority::Low => write!(f, "low"),
            DiffPriority::None => write!(f, "none"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiffMode {
    Strict,
    Relaxed,
    Permissive,
    Custom,
}

impl fmt::Display for DiffMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffMode::Strict => write!(f, "strict"),
            DiffMode::Relaxed => write!(f, "relaxed"),
            DiffMode::Permissive => write!(f, "permissive"),
            DiffMode::Custom => write!(f, "custom"),
        }
    }
}

pub fn diff_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn diff_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = diff_mean(data);
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn diff_std_dev(data: &[f64]) -> f64 {
    diff_variance(data).sqrt()
}

pub fn diff_median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n/2 - 1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
}

/// Percentile calculator for Diff.
pub fn diff_percentile_at(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (hi as f64 - idx) + sorted[hi] * (idx - lo as f64) }
}

pub fn diff_entropy(data: &[f64]) -> f64 {
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

pub fn diff_gini(data: &[f64]) -> f64 {
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

pub fn diff_covariance(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let x: Vec<f64> = data[..n].to_vec();
    let y: Vec<f64> = data[n..2*n].to_vec();
    let mx = diff_mean(&x);
    let my = diff_mean(&y);
    x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - mx) * (yi - my)).sum::<f64>() / (n - 1) as f64
}

pub fn diff_correlation(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let n = data.len() / 2;
    let cov = diff_covariance(data);
    let sx = diff_std_dev(&data[..n]);
    let sy = diff_std_dev(&data[n..2*n]);
    if sx * sy == 0.0 { 0.0 } else { cov / (sx * sy) }
}

pub fn diff_excess_kurtosis(data: &[f64]) -> f64 {
    if data.len() < 4 { return 0.0; }
    let m = diff_mean(data);
    let s = diff_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    let k = data.iter().map(|&x| ((x - m) / s).powi(4)).sum::<f64>() / n;
    k - 3.0
}

pub fn diff_sample_skewness(data: &[f64]) -> f64 {
    if data.len() < 3 { return 0.0; }
    let m = diff_mean(data);
    let s = diff_std_dev(data);
    if s == 0.0 { return 0.0; }
    let n = data.len() as f64;
    data.iter().map(|&x| ((x - m) / s).powi(3)).sum::<f64>() / n
}

pub fn diff_harmmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    n / data.iter().map(|&x| 1.0 / x).sum::<f64>()
}

pub fn diff_geomean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) { return 0.0; }
    let n = data.len() as f64;
    (data.iter().map(|&x| x.ln()).sum::<f64>() / n).exp()
}

/// Iterator over diff analysis results.
#[derive(Debug, Clone)]
pub struct DiffResultIterator {
    items: Vec<(usize, f64)>,
    position: usize,
}

impl DiffResultIterator {
    pub fn new(items: Vec<(usize, f64)>) -> Self {
        DiffResultIterator { items, position: 0 }
    }
    pub fn remaining(&self) -> usize { self.items.len() - self.position }
}

impl Iterator for DiffResultIterator {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.items.len() {
            let item = self.items[self.position];
            self.position += 1;
            Some(item)
        } else { None }
    }
}

/// Convert DiffMerge description to a summary string.
pub fn diffmerge_to_summary(item: &DiffMerge) -> String {
    format!("DiffMerge: {:?}", item)
}

/// Convert ThreeWayDiff description to a summary string.
pub fn threewaydiff_to_summary(item: &ThreeWayDiff) -> String {
    format!("ThreeWayDiff: {:?}", item)
}

/// Convert DiffCompression description to a summary string.
pub fn diffcompression_to_summary(item: &DiffCompression) -> String {
    format!("DiffCompression: {:?}", item)
}

/// Convert ModelComposition description to a summary string.
pub fn modelcomposition_to_summary(item: &ModelComposition) -> String {
    format!("ModelComposition: {:?}", item)
}

/// Convert ModelWeakening description to a summary string.
pub fn modelweakening_to_summary(item: &ModelWeakening) -> String {
    format!("ModelWeakening: {:?}", item)
}

/// Convert ModelStrengthening description to a summary string.
pub fn modelstrengthening_to_summary(item: &ModelStrengthening) -> String {
    format!("ModelStrengthening: {:?}", item)
}

/// Convert LitmusTransformation description to a summary string.
pub fn litmustransformation_to_summary(item: &LitmusTransformation) -> String {
    format!("LitmusTransformation: {:?}", item)
}

/// Convert OutcomePrediction description to a summary string.
pub fn outcomeprediction_to_summary(item: &OutcomePrediction) -> String {
    format!("OutcomePrediction: {:?}", item)
}

/// Convert BehaviorCounter description to a summary string.
pub fn behaviorcounter_to_summary(item: &BehaviorCounter) -> String {
    format!("BehaviorCounter: {:?}", item)
}

/// Convert ModelFeatureVector description to a summary string.
pub fn modelfeaturevector_to_summary(item: &ModelFeatureVector) -> String {
    format!("ModelFeatureVector: {:?}", item)
}

/// Convert DiffPatch description to a summary string.
pub fn diffpatch_to_summary(item: &DiffPatch) -> String {
    format!("DiffPatch: {:?}", item)
}

/// Batch processor for diff operations.
#[derive(Debug, Clone)]
pub struct DiffBatchProcessor {
    pub batch_size: usize,
    pub processed: usize,
    pub errors: Vec<String>,
    pub results: Vec<f64>,
}

impl DiffBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        DiffBatchProcessor { batch_size, processed: 0, errors: Vec::new(), results: Vec::new() }
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

impl fmt::Display for DiffBatchProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffBatch(processed={}, errors={})", self.processed, self.errors.len())
    }
}

/// Detailed report for diff analysis.
#[derive(Debug, Clone)]
pub struct DiffReport {
    pub title: String,
    pub sections: Vec<(String, Vec<String>)>,
    pub metrics: Vec<(String, f64)>,
    pub warnings: Vec<String>,
    pub timestamp: u64,
}

impl DiffReport {
    pub fn new(title: impl Into<String>) -> Self {
        DiffReport { title: title.into(), sections: Vec::new(), metrics: Vec::new(), warnings: Vec::new(), timestamp: 0 }
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

impl fmt::Display for DiffReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffReport({})", self.title)
    }
}

/// Configuration for diff analysis.
#[derive(Debug, Clone)]
pub struct DiffConfig {
    pub verbose: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub timeout_ms: u64,
    pub parallel: bool,
    pub output_format: String,
}

impl DiffConfig {
    pub fn default_config() -> Self {
        DiffConfig {
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

impl fmt::Display for DiffConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffConfig(iters={}, tol={:.0e})", self.max_iterations, self.tolerance)
    }
}

/// Histogram for diff data distribution.
#[derive(Debug, Clone)]
pub struct DiffHistogramExt {
    pub bins: Vec<usize>,
    pub bin_edges: Vec<f64>,
    pub total_count: usize,
}

impl DiffHistogramExt {
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return DiffHistogramExt { bins: Vec::new(), bin_edges: Vec::new(), total_count: 0 };
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
        DiffHistogramExt { bins, bin_edges, total_count: data.len() }
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

impl fmt::Display for DiffHistogramExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Histogram(bins={}, total={})", self.num_bins(), self.total_count)
    }
}

/// Adjacency/weight matrix for diff graph analysis.
#[derive(Debug, Clone)]
pub struct DiffGraph {
    pub adjacency: Vec<Vec<bool>>,
    pub weights: Vec<Vec<f64>>,
    pub node_count: usize,
    pub edge_count: usize,
    pub node_labels: Vec<String>,
}

impl DiffGraph {
    pub fn new(n: usize) -> Self {
        DiffGraph {
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
        fn dfs_cycle_diff(v: usize, adj: &[Vec<bool>], visited: &mut [u8]) -> bool {
            visited[v] = 1;
            for w in 0..adj.len() { if adj[v][w] {
                if visited[w] == 1 { return true; }
                if visited[w] == 0 && dfs_cycle_diff(w, adj, visited) { return true; }
            }}
            visited[v] = 2; false
        }
        for i in 0..n {
            if visited[i] == 0 && dfs_cycle_diff(i, &self.adjacency, &mut visited) { return false; }
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

impl fmt::Display for DiffGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffGraph(n={}, e={})", self.node_count, self.edge_count)
    }
}

/// Cache for diff computation results.
#[derive(Debug, Clone)]
pub struct DiffCache {
    entries: Vec<(u64, Vec<f64>)>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl DiffCache {
    pub fn new(capacity: usize) -> Self {
        DiffCache { entries: Vec::new(), capacity, hits: 0, misses: 0 }
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

impl fmt::Display for DiffCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache(size={}, hit_rate={:.1}%)", self.size(), self.hit_rate() * 100.0)
    }
}

/// Compute pairwise distances for diff elements.
pub fn diff_pairwise_distances(points: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// K-means clustering for diff data.
pub fn diff_kmeans(data: &[Vec<f64>], k: usize, max_iters: usize) -> Vec<usize> {
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

/// Principal component analysis (simplified) for diff data.
pub fn diff_pca_2d(data: &[Vec<f64>]) -> Vec<(f64, f64)> {
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

/// Dense matrix operations for Diff computations.
#[derive(Debug, Clone)]
pub struct DiffDenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DiffDenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        DiffDenseMatrix { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        DiffDenseMatrix { rows, cols, data }
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
        DiffDenseMatrix { rows: self.rows, cols: self.cols, data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows); assert_eq!(self.cols, other.cols);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
        DiffDenseMatrix { rows: self.rows, cols: self.cols, data }
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
        DiffDenseMatrix { rows: self.rows, cols: self.cols, data }
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
        DiffDenseMatrix { rows: self.rows, cols: self.cols, data }
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

impl fmt::Display for DiffDenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DiffMatrix({}x{})", self.rows, self.cols)
    }
}

/// Interval arithmetic for Diff bounds analysis.
#[derive(Debug, Clone, Copy)]
pub struct DiffInterval {
    pub lo: f64,
    pub hi: f64,
}

impl DiffInterval {
    pub fn new(lo: f64, hi: f64) -> Self {
        DiffInterval { lo: lo.min(hi), hi: lo.max(hi) }
    }

    pub fn point(v: f64) -> Self {
        DiffInterval { lo: v, hi: v }
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
        DiffInterval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(DiffInterval { lo, hi }) } else { None }
    }

    pub fn add(&self, other: &Self) -> Self {
        DiffInterval { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn sub(&self, other: &Self) -> Self {
        DiffInterval { lo: self.lo - other.hi, hi: self.hi - other.lo }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        DiffInterval { lo, hi }
    }

    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 { *self }
        else if self.hi <= 0.0 { DiffInterval { lo: -self.hi, hi: -self.lo } }
        else { DiffInterval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) } }
    }

    pub fn sqrt(&self) -> Self {
        let lo = if self.lo >= 0.0 { self.lo.sqrt() } else { 0.0 };
        DiffInterval { lo, hi: self.hi.max(0.0).sqrt() }
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

impl fmt::Display for DiffInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

/// State machine for Diff protocol modeling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DiffState {
    Empty,
    Loaded,
    Diffing,
    Merged,
    Applied,
    Reverted,
}

impl fmt::Display for DiffState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffState::Empty => write!(f, "empty"),
            DiffState::Loaded => write!(f, "loaded"),
            DiffState::Diffing => write!(f, "diffing"),
            DiffState::Merged => write!(f, "merged"),
            DiffState::Applied => write!(f, "applied"),
            DiffState::Reverted => write!(f, "reverted"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiffStateMachine {
    pub current: DiffState,
    pub history: Vec<String>,
    pub transition_count: usize,
}

impl DiffStateMachine {
    pub fn new() -> Self {
        DiffStateMachine { current: DiffState::Empty, history: Vec::new(), transition_count: 0 }
    }
    pub fn state(&self) -> &DiffState { &self.current }
    pub fn can_transition(&self, target: &DiffState) -> bool {
        match (&self.current, target) {
            (DiffState::Empty, DiffState::Loaded) => true,
            (DiffState::Loaded, DiffState::Diffing) => true,
            (DiffState::Diffing, DiffState::Merged) => true,
            (DiffState::Merged, DiffState::Applied) => true,
            (DiffState::Applied, DiffState::Reverted) => true,
            (DiffState::Reverted, DiffState::Loaded) => true,
            (DiffState::Merged, DiffState::Reverted) => true,
            _ => false,
        }
    }
    pub fn transition(&mut self, target: DiffState) -> bool {
        if self.can_transition(&target) {
            self.history.push(format!("{} -> {}", self.current, target));
            self.current = target;
            self.transition_count += 1;
            true
        } else { false }
    }
    pub fn reset(&mut self) {
        self.current = DiffState::Empty;
        self.history.clear();
        self.transition_count = 0;
    }
    pub fn history_len(&self) -> usize { self.history.len() }
}

impl fmt::Display for DiffStateMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SM(state={}, transitions={})", self.current, self.transition_count)
    }
}

/// Ring buffer for Diff event tracking.
#[derive(Debug, Clone)]
pub struct DiffRingBuffer {
    data: Vec<f64>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl DiffRingBuffer {
    pub fn new(capacity: usize) -> Self {
        DiffRingBuffer { data: vec![0.0; capacity], capacity, head: 0, count: 0 }
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

impl fmt::Display for DiffRingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RingBuffer(len={}/{})", self.count, self.capacity)
    }
}

/// Disjoint set (union-find) for Diff component tracking.
#[derive(Debug, Clone)]
pub struct DiffDisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    num_components: usize,
}

impl DiffDisjointSet {
    pub fn new(n: usize) -> Self {
        DiffDisjointSet { parent: (0..n).collect(), rank: vec![0; n], size: vec![1; n], num_components: n }
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

impl fmt::Display for DiffDisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet(n={}, components={})", self.parent.len(), self.num_components)
    }
}

/// Sorted list with binary search for Diff.
#[derive(Debug, Clone)]
pub struct DiffSortedList {
    data: Vec<f64>,
}

impl DiffSortedList {
    pub fn new() -> Self { DiffSortedList { data: Vec::new() } }
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

impl fmt::Display for DiffSortedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SortedList(len={})", self.data.len())
    }
}

/// Exponential moving average for Diff metrics.
#[derive(Debug, Clone)]
pub struct DiffEma {
    pub alpha: f64,
    pub value: f64,
    pub count: usize,
    pub initialized: bool,
}

impl DiffEma {
    pub fn new(alpha: f64) -> Self { DiffEma { alpha, value: 0.0, count: 0, initialized: false } }
    pub fn update(&mut self, sample: f64) {
        if !self.initialized { self.value = sample; self.initialized = true; }
        else { self.value = self.alpha * sample + (1.0 - self.alpha) * self.value; }
        self.count += 1;
    }
    pub fn current(&self) -> f64 { self.value }
    pub fn reset(&mut self) { self.value = 0.0; self.count = 0; self.initialized = false; }
}

impl fmt::Display for DiffEma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EMA(alpha={:.2}, value={:.4})", self.alpha, self.value)
    }
}

/// Simple bloom filter for Diff membership testing.
#[derive(Debug, Clone)]
pub struct DiffBloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl DiffBloomFilter {
    pub fn new(size: usize, num_hashes: usize) -> Self {
        DiffBloomFilter { bits: vec![false; size], num_hashes, size, count: 0 }
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

impl fmt::Display for DiffBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BloomFilter(size={}, count={}, fpr={:.4})", self.size, self.count, self.false_positive_rate())
    }
}

/// Simple prefix trie for Diff string matching.
#[derive(Debug, Clone)]
pub struct DiffTrieNode {
    children: Vec<(char, usize)>,
    is_terminal: bool,
    value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct DiffTrie {
    nodes: Vec<DiffTrieNode>,
    count: usize,
}

impl DiffTrie {
    pub fn new() -> Self {
        DiffTrie { nodes: vec![DiffTrieNode { children: Vec::new(), is_terminal: false, value: None }], count: 0 }
    }
    pub fn insert(&mut self, key: &str, value: u64) {
        let mut current = 0;
        for ch in key.chars() {
            let next = self.nodes[current].children.iter().find(|(c, _)| *c == ch).map(|(_, idx)| *idx);
            current = match next {
                Some(idx) => idx,
                None => {
                    let idx = self.nodes.len();
                    self.nodes.push(DiffTrieNode { children: Vec::new(), is_terminal: false, value: None });
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

impl fmt::Display for DiffTrie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie(entries={}, nodes={})", self.count, self.nodes.len())
    }
}

/// Min-heap priority queue for Diff scheduling.
#[derive(Debug, Clone)]
pub struct DiffPriorityQueue {
    heap: Vec<(f64, usize)>,
}

impl DiffPriorityQueue {
    pub fn new() -> Self { DiffPriorityQueue { heap: Vec::new() } }
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

impl fmt::Display for DiffPriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQ(len={})", self.heap.len())
    }
}

/// Online statistics accumulator for Diff.
#[derive(Debug, Clone)]
pub struct DiffAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    min_val: f64,
    max_val: f64,
    sum: f64,
}

impl DiffAccumulator {
    pub fn new() -> Self { DiffAccumulator { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
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

impl fmt::Display for DiffAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Acc(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Sparse matrix (COO format) for Diff.
#[derive(Debug, Clone)]
pub struct DiffSparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

impl DiffSparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self { DiffSparseMatrix { rows, cols, entries: Vec::new() } }
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
        let mut result = DiffSparseMatrix::new(self.cols, self.rows);
        for &(i, j, v) in &self.entries { result.entries.push((j, i, v)); }
        result
    }
    pub fn add(&self, other: &Self) -> Self {
        let mut result = DiffSparseMatrix::new(self.rows, self.cols);
        for &(i, j, v) in &self.entries { result.insert(i, j, result.get(i, j) + v); }
        for &(i, j, v) in &other.entries { result.insert(i, j, result.get(i, j) + v); }
        result
    }
    pub fn scale(&self, s: f64) -> Self {
        let mut result = DiffSparseMatrix::new(self.rows, self.cols);
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

impl fmt::Display for DiffSparseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sparse({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

/// Polynomial with f64 coefficients for Diff.
#[derive(Debug, Clone)]
pub struct DiffPolynomial {
    pub coefficients: Vec<f64>,
}

impl DiffPolynomial {
    pub fn new(coeffs: Vec<f64>) -> Self { DiffPolynomial { coefficients: coeffs } }
    pub fn zero() -> Self { DiffPolynomial { coefficients: vec![0.0] } }
    pub fn one() -> Self { DiffPolynomial { coefficients: vec![1.0] } }
    pub fn monomial(degree: usize, coeff: f64) -> Self {
        let mut c = vec![0.0; degree + 1];
        c[degree] = coeff;
        DiffPolynomial { coefficients: c }
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
        DiffPolynomial { coefficients: result }
    }
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coefficients.len().max(other.coefficients.len());
        let mut result = vec![0.0; n];
        for (i, &c) in self.coefficients.iter().enumerate() { result[i] += c; }
        for (i, &c) in other.coefficients.iter().enumerate() { result[i] -= c; }
        DiffPolynomial { coefficients: result }
    }
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.coefficients.len() + other.coefficients.len() - 1;
        let mut result = vec![0.0; n];
        for (i, &a) in self.coefficients.iter().enumerate() {
            for (j, &b) in other.coefficients.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        DiffPolynomial { coefficients: result }
    }
    pub fn scale(&self, s: f64) -> Self {
        DiffPolynomial { coefficients: self.coefficients.iter().map(|&c| c * s).collect() }
    }
    pub fn derivative(&self) -> Self {
        if self.coefficients.len() <= 1 { return Self::zero(); }
        let coeffs: Vec<f64> = self.coefficients.iter().enumerate().skip(1)
            .map(|(i, &c)| c * i as f64).collect();
        DiffPolynomial { coefficients: coeffs }
    }
    pub fn integral(&self, constant: f64) -> Self {
        let mut coeffs = vec![constant];
        for (i, &c) in self.coefficients.iter().enumerate() {
            coeffs.push(c / (i + 1) as f64);
        }
        DiffPolynomial { coefficients: coeffs }
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

impl fmt::Display for DiffPolynomial {
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

/// Simple linear congruential generator for Diff.
#[derive(Debug, Clone)]
pub struct DiffRng {
    state: u64,
}

impl DiffRng {
    pub fn new(seed: u64) -> Self { DiffRng { state: seed } }
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

impl fmt::Display for DiffRng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rng(state={:#x})", self.state)
    }
}

/// Simple timing utility for Diff benchmarking.
#[derive(Debug, Clone)]
pub struct DiffTimer {
    pub label: String,
    pub elapsed_ns: Vec<u64>,
    pub running: bool,
}

impl DiffTimer {
    pub fn new(label: impl Into<String>) -> Self { DiffTimer { label: label.into(), elapsed_ns: Vec::new(), running: false } }
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

impl fmt::Display for DiffTimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timer({}: avg={:.0}ns, n={})", self.label, self.average_ns(), self.count())
    }
}

/// Compact bit vector for Diff set operations.
#[derive(Debug, Clone)]
pub struct DiffBitVector {
    words: Vec<u64>,
    len: usize,
}

impl DiffBitVector {
    pub fn new(len: usize) -> Self { DiffBitVector { words: vec![0u64; (len + 63) / 64], len } }
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

impl fmt::Display for DiffBitVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec(len={}, ones={})", self.len, self.count_ones())
    }
}

/// LRU cache for Diff computation memoization.
#[derive(Debug, Clone)]
pub struct DiffLruCache {
    entries: Vec<(u64, Vec<f64>, u64)>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl DiffLruCache {
    pub fn new(capacity: usize) -> Self { DiffLruCache { entries: Vec::new(), capacity, clock: 0, hits: 0, misses: 0, evictions: 0 } }
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

impl fmt::Display for DiffLruCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRU(size={}/{}, hr={:.1}%)", self.size(), self.capacity, self.hit_rate() * 100.0)
    }
}

/// Graph coloring utility for Diff scheduling.
#[derive(Debug, Clone)]
pub struct DiffGraphColoring {
    pub adjacency: Vec<Vec<bool>>,
    pub colors: Vec<Option<usize>>,
    pub num_nodes: usize,
    pub num_colors_used: usize,
}

impl DiffGraphColoring {
    pub fn new(n: usize) -> Self {
        DiffGraphColoring { adjacency: vec![vec![false; n]; n], colors: vec![None; n], num_nodes: n, num_colors_used: 0 }
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

impl fmt::Display for DiffGraphColoring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Coloring(n={}, colors={})", self.num_nodes, self.num_colors_used)
    }
}

/// Top-K tracker for Diff ranking.
#[derive(Debug, Clone)]
pub struct DiffTopK {
    pub k: usize,
    pub items: Vec<(f64, String)>,
}

impl DiffTopK {
    pub fn new(k: usize) -> Self { DiffTopK { k, items: Vec::new() } }
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

impl fmt::Display for DiffTopK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TopK(k={}, len={})", self.k, self.items.len())
    }
}

/// Sliding window statistics for Diff monitoring.
#[derive(Debug, Clone)]
pub struct DiffSlidingWindow {
    pub window_size: usize,
    pub data: Vec<f64>,
    pub sum: f64,
}

impl DiffSlidingWindow {
    pub fn new(window_size: usize) -> Self { DiffSlidingWindow { window_size, data: Vec::new(), sum: 0.0 } }
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

impl fmt::Display for DiffSlidingWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Window(size={}/{}, mean={:.2})", self.data.len(), self.window_size, self.mean())
    }
}

/// Confusion matrix for Diff classification evaluation.
#[derive(Debug, Clone)]
pub struct DiffConfusionMatrix {
    pub true_positive: u64,
    pub false_positive: u64,
    pub true_negative: u64,
    pub false_negative: u64,
}

impl DiffConfusionMatrix {
    pub fn new() -> Self { DiffConfusionMatrix { true_positive: 0, false_positive: 0, true_negative: 0, false_negative: 0 } }
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

impl fmt::Display for DiffConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CM(acc={:.3}, prec={:.3}, rec={:.3}, f1={:.3})",
            self.accuracy(), self.precision(), self.recall(), self.f1_score())
    }
}

/// Cosine similarity for Diff feature vectors.
pub fn diff_cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na * nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Euclidean distance for Diff.
pub fn diff_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Manhattan distance for Diff.
pub fn diff_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Chebyshev distance for Diff.
pub fn diff_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// Minkowski distance for Diff.
pub fn diff_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

/// Normalize a vector for Diff.
pub fn diff_normalize(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 { v.to_vec() } else { v.iter().map(|x| x / norm).collect() }
}

/// Dot product for Diff.
pub fn diff_dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cross product (3D) for Diff.
pub fn diff_cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

/// Linear interpolation for Diff.
pub fn diff_lerp(a: f64, b: f64, t: f64) -> f64 { a + (b - a) * t }

/// Clamp value for Diff.
pub fn diff_clamp(v: f64, lo: f64, hi: f64) -> f64 { v.max(lo).min(hi) }

/// Sigmoid function for Diff.
pub fn diff_sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

/// Softmax for Diff.
pub fn diff_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Log-sum-exp for Diff.
pub fn diff_logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

/// KL divergence for Diff.
pub fn diff_kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        if pi > 0.0 && qi > 0.0 { pi * (pi / qi).ln() } else { 0.0 }
    }).sum()
}

/// Jensen-Shannon divergence for Diff.
pub fn diff_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    (diff_kl_divergence(p, &m) + diff_kl_divergence(q, &m)) / 2.0
}

/// Total variation distance for Diff.
pub fn diff_tv_distance(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi - qi).abs()).sum::<f64>() / 2.0
}

/// Hellinger distance for Diff.
pub fn diff_hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let sum: f64 = p.iter().zip(q.iter()).map(|(&pi, &qi)| {
        let diff = pi.sqrt() - qi.sqrt();
        diff * diff
    }).sum();
    (sum / 2.0).sqrt()
}

/// Earth mover's distance (1D) for Diff.
pub fn diff_emd_1d(p: &[f64], q: &[f64]) -> f64 {
    let mut cum_diff = 0.0;
    let mut total = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        cum_diff += pi - qi;
        total += cum_diff.abs();
    }
    total
}

/// Feature scaling utilities for Diff.
#[derive(Debug, Clone)]
pub struct DiffFeatureScaler {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
    pub fitted: bool,
}

impl DiffFeatureScaler {
    pub fn new() -> Self { DiffFeatureScaler { means: Vec::new(), stds: Vec::new(), mins: Vec::new(), maxs: Vec::new(), fitted: false } }
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

impl fmt::Display for DiffFeatureScaler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scaler(dim={}, fitted={})", self.dimension(), self.fitted)
    }
}

/// Simple linear regression for Diff trend analysis.
#[derive(Debug, Clone)]
pub struct DiffLinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub fitted: bool,
}

impl DiffLinearRegression {
    pub fn new() -> Self { DiffLinearRegression { slope: 0.0, intercept: 0.0, r_squared: 0.0, fitted: false } }
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

impl fmt::Display for DiffLinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R2={:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Weighted undirected graph for Diff.
#[derive(Debug, Clone)]
pub struct DiffWeightedGraph {
    pub adj: Vec<Vec<(usize, f64)>>,
    pub num_nodes: usize,
    pub num_edges: usize,
}

impl DiffWeightedGraph {
    pub fn new(n: usize) -> Self { DiffWeightedGraph { adj: vec![Vec::new(); n], num_nodes: n, num_edges: 0 } }
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
        fn find_diff(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x { parent[x] = find_diff(parent, parent[x]); }
            parent[x]
        }
        let mut total = 0.0;
        let mut count = 0;
        for (w, u, v) in edges {
            let ru = find_diff(&mut parent, u);
            let rv = find_diff(&mut parent, v);
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

impl fmt::Display for DiffWeightedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WGraph(n={}, e={})", self.num_nodes, self.num_edges)
    }
}

/// Moving average for Diff.
pub fn diff_moving_average(data: &[f64], window: usize) -> Vec<f64> {
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

/// Cumulative sum for Diff.
pub fn diff_cumsum(data: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut sum = 0.0;
    for &v in data { sum += v; result.push(sum); }
    result
}

/// Numerical differentiation for Diff.
pub fn diff_diff(data: &[f64]) -> Vec<f64> {
    if data.len() < 2 { return Vec::new(); }
    data.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Auto-correlation for Diff.
pub fn diff_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag { return 0.0; }
    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..(n - lag)).map(|i| (data[i] - mean) * (data[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Discrete Fourier transform magnitude for Diff.
pub fn diff_dft_magnitude(data: &[f64]) -> Vec<f64> {
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

/// Trapezoidal integration for Diff.
pub fn diff_integrate_trapezoid(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let mut total = 0.0;
    for i in 1..x.len() {
        total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0;
    }
    total
}

/// Simpson's rule integration for Diff.
pub fn diff_integrate_simpson(x: &[f64], y: &[f64]) -> f64 {
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

/// Convolution for Diff.
pub fn diff_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Axis-aligned bounding box for Diff spatial indexing.
#[derive(Debug, Clone, Copy)]
pub struct DiffAABB {
    pub x_min: f64, pub y_min: f64,
    pub x_max: f64, pub y_max: f64,
}

impl DiffAABB {
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self { DiffAABB { x_min, y_min, x_max, y_max } }
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
            DiffAABB::new(self.x_min, self.y_min, cx, cy),
            DiffAABB::new(cx, self.y_min, self.x_max, cy),
            DiffAABB::new(self.x_min, cy, cx, self.y_max),
            DiffAABB::new(cx, cy, self.x_max, self.y_max),
        ]
    }
}

/// 2D point for Diff.
#[derive(Debug, Clone, Copy)]
pub struct DiffPoint2D { pub x: f64, pub y: f64, pub data: f64 }

/// Quadtree for Diff spatial indexing.
#[derive(Debug, Clone)]
pub struct DiffQuadTree {
    pub boundary: DiffAABB,
    pub points: Vec<DiffPoint2D>,
    pub children: Option<Vec<DiffQuadTree>>,
    pub capacity: usize,
    pub depth: usize,
    pub max_depth: usize,
}

impl DiffQuadTree {
    pub fn new(boundary: DiffAABB, capacity: usize, max_depth: usize) -> Self {
        DiffQuadTree { boundary, points: Vec::new(), children: None, capacity, depth: 0, max_depth }
    }
    fn with_depth(boundary: DiffAABB, capacity: usize, depth: usize, max_depth: usize) -> Self {
        DiffQuadTree { boundary, points: Vec::new(), children: None, capacity, depth, max_depth }
    }
    pub fn insert(&mut self, p: DiffPoint2D) -> bool {
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
            children.push(DiffQuadTree::with_depth(*q, self.capacity, self.depth + 1, self.max_depth));
        }
        let old_points: Vec<_> = self.points.drain(..).collect();
        self.children = Some(children);
        for p in old_points { self.insert(p); }
    }
    pub fn query_range(&self, range: &DiffAABB) -> Vec<DiffPoint2D> {
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

impl fmt::Display for DiffQuadTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTree(count={}, depth={})", self.count(), self.tree_depth())
    }
}

/// QR decomposition helper for Diff.
pub fn diff_qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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

/// Solve upper triangular system Rx = b for Diff.
pub fn diff_solve_upper_triangular(r: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i+1)..n { s -= r[i][j] * x[j]; }
        x[i] = if r[i][i].abs() > 1e-15 { s / r[i][i] } else { 0.0 };
    }
    x
}

/// Matrix-vector multiply for Diff.
pub fn diff_mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum()).collect()
}

/// Matrix transpose for Diff.
pub fn diff_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() { return Vec::new(); }
    let m = a.len(); let n = a[0].len();
    (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect()
}

/// Matrix multiply for Diff.
pub fn diff_mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 { return Vec::new(); }
    let k = a[0].len();
    let n = b[0].len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m { for j in 0..n { for l in 0..k { c[i][j] += a[i][l] * b[l][j]; } } }
    c
}

/// Frobenius norm for Diff.
pub fn diff_frobenius_norm(a: &[Vec<f64>]) -> f64 {
    a.iter().flat_map(|row| row.iter()).map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix trace for Diff.
pub fn diff_trace(a: &[Vec<f64>]) -> f64 {
    a.iter().enumerate().map(|(i, row)| if i < row.len() { row[i] } else { 0.0 }).sum()
}

/// Identity matrix for Diff.
pub fn diff_identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n { m[i][i] = 1.0; }
    m
}

/// Power iteration for dominant eigenvalue for Diff.
pub fn diff_power_iteration(a: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = a.len();
    let mut v = vec![1.0; n];
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in &mut v { *x /= norm; }
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = diff_mat_vec_mul(a, &v);
        let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm.abs() < 1e-15 { break; }
        eigenvalue = av.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();
        v = av.iter().map(|&x| x / norm).collect();
    }
    (eigenvalue, v)
}

/// Running statistics with min/max tracking for Diff.
#[derive(Debug, Clone)]
pub struct DiffRunningStats {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub min_val: f64,
    pub max_val: f64,
    pub sum: f64,
}

impl DiffRunningStats {
    pub fn new() -> Self { DiffRunningStats { count: 0, mean: 0.0, m2: 0.0, min_val: f64::INFINITY, max_val: f64::NEG_INFINITY, sum: 0.0 } }
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

impl fmt::Display for DiffRunningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(n={}, mean={:.4}, std={:.4})", self.count, self.mean, self.std_dev())
    }
}

/// Interquartile range for Diff.
pub fn diff_iqr(data: &[f64]) -> f64 {
    diff_percentile_at(data, 75.0) - diff_percentile_at(data, 25.0)
}

/// Detect outliers using IQR method for Diff.
pub fn diff_outliers(data: &[f64]) -> Vec<usize> {
    let q1 = diff_percentile_at(data, 25.0);
    let q3 = diff_percentile_at(data, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    data.iter().enumerate().filter(|(_, &v)| v < lower || v > upper).map(|(i, _)| i).collect()
}

/// Z-score normalization for Diff.
pub fn diff_zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    if n < 2.0 { return data.to_vec(); }
    let mean = data.iter().sum::<f64>() / n;
    let std = (data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    if std.abs() < 1e-15 { return vec![0.0; data.len()]; }
    data.iter().map(|&x| (x - mean) / std).collect()
}

/// Rank values for Diff.
pub fn diff_rank(data: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut ranks = vec![0.0; data.len()];
    for (rank, &(idx, _)) in indexed.iter().enumerate() { ranks[idx] = (rank + 1) as f64; }
    ranks
}

/// Spearman rank correlation for Diff.
pub fn diff_spearman(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rx = diff_rank(x);
    let ry = diff_rank(y);
    let n = x.len() as f64;
    let d_sq: f64 = rx.iter().zip(ry.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
    1.0 - 6.0 * d_sq / (n * (n * n - 1.0))
}

/// Covariance matrix for Diff.
pub fn diff_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

/// Correlation matrix for Diff.
pub fn diff_correlation_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = diff_covariance_matrix(data);
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

    fn make_model(name: &str, axiom_names: &[&str], rel_names: &[&str]) -> MemoryModelSpec {
        let mut model = MemoryModelSpec::new(name);
        for &an in axiom_names {
            model.add_axiom(AxiomSpec::new(an, AxiomKind::Acyclicity));
        }
        for &rn in rel_names {
            model.add_relation(RelationSpec::base(rn, RelationDef::Base));
        }
        model
    }

    fn make_behavior_set(name: &str, behaviors: &[(&str, u64)]) -> BehaviorSet {
        let mut set = BehaviorSet::new(name);
        for (var, val) in behaviors {
            set.add(Behavior::new("test").with_final(*var, *val));
        }
        set
    }

    #[test]
    fn test_model_spec_basic() {
        let model = make_model("SC", &["hb-acyclic", "mo-total"], &["po", "rf", "mo"]);
        assert_eq!(model.axioms.len(), 2);
        assert_eq!(model.relations.len(), 3);
        assert!(model.find_axiom("hb-acyclic").is_some());
        assert!(model.find_relation("po").is_some());
    }

    #[test]
    fn test_relation_def_display() {
        let def = RelationDef::Union(vec!["po".into(), "rf".into()]);
        assert_eq!(format!("{}", def), "(po | rf)");

        let def2 = RelationDef::TransitiveClosure("hb".into());
        assert_eq!(format!("{}", def2), "hb+");
    }

    #[test]
    fn test_behavior_fingerprint() {
        let b1 = Behavior::new("t1").with_final("x", 1);
        let b2 = Behavior::new("t1").with_final("x", 1);
        let b3 = Behavior::new("t1").with_final("x", 2);
        assert_eq!(b1.fingerprint(), b2.fingerprint());
        assert_ne!(b1.fingerprint(), b3.fingerprint());
    }

    #[test]
    fn test_behavior_set_operations() {
        let mut s1 = BehaviorSet::new("M1");
        s1.add(Behavior::new("t").with_final("x", 1));
        s1.add(Behavior::new("t").with_final("x", 2));

        let mut s2 = BehaviorSet::new("M2");
        s2.add(Behavior::new("t").with_final("x", 2));
        s2.add(Behavior::new("t").with_final("x", 3));

        assert_eq!(s1.union(&s2).len(), 3);
        assert_eq!(s1.intersection(&s2).len(), 1);
        assert_eq!(s1.difference(&s2).len(), 1);
    }

    #[test]
    fn test_behavior_set_subset() {
        let mut s1 = BehaviorSet::new("M1");
        s1.add(Behavior::new("t").with_final("x", 1));

        let mut s2 = BehaviorSet::new("M2");
        s2.add(Behavior::new("t").with_final("x", 1));
        s2.add(Behavior::new("t").with_final("x", 2));

        assert!(s1.is_subset_of(&s2));
        assert!(!s2.is_subset_of(&s1));
        assert!(s2.is_superset_of(&s1));
    }

    #[test]
    fn test_compute_diff_identical() {
        let m1 = make_model("SC", &["hb-acyclic"], &["po", "rf"]);
        let m2 = make_model("SC2", &["hb-acyclic"], &["po", "rf"]);
        let diff = compute_diff(&m1, &m2);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_compute_diff_added_axiom() {
        let m1 = make_model("M1", &["hb-acyclic"], &["po"]);
        let m2 = make_model("M2", &["hb-acyclic", "mo-total"], &["po"]);
        let diff = compute_diff(&m1, &m2);
        assert_eq!(diff.additions().len(), 1);
        assert_eq!(diff.removals().len(), 0);
    }

    #[test]
    fn test_compute_diff_removed_relation() {
        let m1 = make_model("M1", &["hb-acyclic"], &["po", "rf"]);
        let m2 = make_model("M2", &["hb-acyclic"], &["po"]);
        let diff = compute_diff(&m1, &m2);
        assert_eq!(diff.removals().len(), 1);
    }

    #[test]
    fn test_diff_summary() {
        let m1 = make_model("M1", &["a1", "a2"], &["r1"]);
        let m2 = make_model("M2", &["a1", "a3"], &["r1", "r2"]);
        let diff = compute_diff(&m1, &m2);
        let summary = diff.summary();
        assert!(summary.total_changes > 0);
        assert!(!summary.is_identical());
    }

    #[test]
    fn test_diff_matrix() {
        let models = vec![
            make_model("SC", &["hb-acyclic", "mo-total"], &["po", "rf", "mo"]),
            make_model("TSO", &["hb-acyclic"], &["po", "rf", "mo", "wb"]),
            make_model("Relaxed", &[], &["po", "rf"]),
        ];
        let matrix = diff_matrix(&models);
        assert_eq!(matrix.size(), 3);
        assert_eq!(matrix.get(0, 0).total_changes, 0);
        assert!(matrix.get(0, 2).total_changes > 0);
    }

    #[test]
    fn test_subset_checker() {
        let mut s1 = BehaviorSet::new("SC");
        s1.add(Behavior::new("t").with_final("x", 1));

        let mut s2 = BehaviorSet::new("TSO");
        s2.add(Behavior::new("t").with_final("x", 1));
        s2.add(Behavior::new("t").with_final("x", 2));

        let result = SubsetChecker::is_behavioral_subset(&s1, &s2);
        assert!(result.is_subset);
        assert!(result.is_strict_subset());
        assert_eq!(result.counterexamples.len(), 0);
    }

    #[test]
    fn test_subset_checker_not_subset() {
        let mut s1 = BehaviorSet::new("M1");
        s1.add(Behavior::new("t").with_final("x", 1));
        s1.add(Behavior::new("t").with_final("x", 99));

        let mut s2 = BehaviorSet::new("M2");
        s2.add(Behavior::new("t").with_final("x", 1));

        let result = SubsetChecker::is_behavioral_subset(&s1, &s2);
        assert!(!result.is_subset);
        assert_eq!(result.counterexamples.len(), 1);
    }

    #[test]
    fn test_refinement_checker() {
        let mut spec = BehaviorSet::new("Spec");
        spec.add(Behavior::new("t").with_final("x", 1));
        spec.add(Behavior::new("t").with_final("x", 2));

        let mut impl_ = BehaviorSet::new("Impl");
        impl_.add(Behavior::new("t").with_final("x", 1));

        let result = RefinementChecker::is_refinement(&spec, &impl_);
        assert!(result.is_refinement);
        assert_eq!(result.coverage, 0.5);
    }

    #[test]
    fn test_refinement_violation() {
        let mut spec = BehaviorSet::new("Spec");
        spec.add(Behavior::new("t").with_final("x", 1));

        let mut impl_ = BehaviorSet::new("Impl");
        impl_.add(Behavior::new("t").with_final("x", 1));
        impl_.add(Behavior::new("t").with_final("x", 99));

        let result = RefinementChecker::is_refinement(&spec, &impl_);
        assert!(!result.is_refinement);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_find_discriminators() {
        let mut sc = BehaviorSet::new("SC");
        sc.add(Behavior::new("mp").with_final("x", 1));

        let mut tso = BehaviorSet::new("TSO");
        tso.add(Behavior::new("mp").with_final("x", 1));
        tso.add(Behavior::new("mp").with_final("x", 0));

        let models = vec![("SC", &sc), ("TSO", &tso)];
        let disc_set = find_discriminators(&models);
        assert!(!disc_set.is_empty());
        let pair = disc_set.for_pair("SC", "TSO");
        assert!(!pair.is_empty());
    }

    #[test]
    fn test_minimal_discriminator() {
        let mut sc = BehaviorSet::new("SC");
        sc.add(Behavior::new("mp").with_final("x", 1));

        let mut tso = BehaviorSet::new("TSO");
        tso.add(Behavior::new("mp").with_final("x", 1));
        tso.add(Behavior::new("mp").with_final("x", 0));

        let models = vec![("SC", &sc), ("TSO", &tso)];
        let disc_set = find_discriminators(&models);
        let min = minimal_discriminator("SC", "TSO", &disc_set);
        assert!(min.is_some());
    }

    #[test]
    fn test_jaccard_distance() {
        let mut s1 = BehaviorSet::new("M1");
        s1.add(Behavior::new("t").with_final("x", 1));

        let mut s2 = BehaviorSet::new("M2");
        s2.add(Behavior::new("t").with_final("x", 1));

        let d = ModelDistanceExt::jaccard_distance(&s1, &s2);
        assert!((d - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_distance_disjoint() {
        let mut s1 = BehaviorSet::new("M1");
        s1.add(Behavior::new("t").with_final("x", 1));

        let mut s2 = BehaviorSet::new("M2");
        s2.add(Behavior::new("t").with_final("x", 2));

        let d = ModelDistanceExt::jaccard_distance(&s1, &s2);
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hamming_distance() {
        let mut s1 = BehaviorSet::new("M1");
        s1.add(Behavior::new("t").with_final("x", 1));
        s1.add(Behavior::new("t").with_final("x", 2));

        let mut s2 = BehaviorSet::new("M2");
        s2.add(Behavior::new("t").with_final("x", 2));
        s2.add(Behavior::new("t").with_final("x", 3));

        let d = ModelDistanceExt::hamming_distance(&s1, &s2);
        assert!((d - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_matrix() {
        let mut s1 = BehaviorSet::new("SC");
        s1.add(Behavior::new("t").with_final("x", 1));
        let mut s2 = BehaviorSet::new("TSO");
        s2.add(Behavior::new("t").with_final("x", 1));
        s2.add(Behavior::new("t").with_final("x", 2));

        let sets = vec![("SC", &s1), ("TSO", &s2)];
        let dm = distance_matrix(&sets, ModelDistanceExt::jaccard_distance);
        assert_eq!(dm.distances[0][0], 0.0);
        assert!(dm.distances[0][1] > 0.0);
        assert_eq!(dm.distances[0][1], dm.distances[1][0]);
    }

    #[test]
    fn test_hierarchical_clustering() {
        let mut s1 = BehaviorSet::new("A");
        s1.add(Behavior::new("t").with_final("x", 1));
        let mut s2 = BehaviorSet::new("B");
        s2.add(Behavior::new("t").with_final("x", 1));
        let mut s3 = BehaviorSet::new("C");
        s3.add(Behavior::new("t").with_final("x", 2));

        let sets = vec![("A", &s1), ("B", &s2), ("C", &s3)];
        let dm = distance_matrix(&sets, ModelDistanceExt::jaccard_distance);
        let clusters = dm.hierarchical_clustering();
        assert!(!clusters.is_empty());
        // A and B should merge first (distance 0)
        assert_eq!(clusters[0].2, 0.0);
    }

    #[test]
    fn test_diff_formatter() {
        let m1 = make_model("SC", &["hb-acyclic"], &["po", "rf"]);
        let m2 = make_model("TSO", &["hb-acyclic", "sc-total"], &["po", "rf", "wb"]);
        let diff = compute_diff(&m1, &m2);
        let table = DiffFormatter::format_diff_table(&diff);
        assert!(table.contains("SC"));
        assert!(table.contains("TSO"));
    }

    #[test]
    fn test_venn_diagram_formatter() {
        let mut s1 = BehaviorSet::new("SC");
        s1.add(Behavior::new("t").with_final("x", 1));
        let mut s2 = BehaviorSet::new("TSO");
        s2.add(Behavior::new("t").with_final("x", 1));
        s2.add(Behavior::new("t").with_final("x", 2));

        let venn = DiffFormatter::format_venn_diagram(&s1, &s2);
        assert!(venn.contains("SC"));
        assert!(venn.contains("TSO"));
        assert!(venn.contains("Common"));
    }

    #[test]
    fn test_constraint_spec() {
        let c = ConstraintSpec::new("coh", ConstraintKind::Coherence)
            .with_description("Per-location coherence");
        assert_eq!(format!("{}", c.kind), "coherence");
    }

    #[test]
    fn test_model_complexity_score() {
        let m = make_model("M", &["a1", "a2"], &["r1", "r2", "r3"]);
        assert_eq!(m.complexity_score(), 2 + 6);
    }
    #[test]
    fn test_diffmerge_new() {
        let item = DiffMerge::new(Vec::new(), Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_threewaydiff_new() {
        let item = ThreeWayDiff::new(Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_diffcompression_new() {
        let item = DiffCompression::new(0, 0, 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_modelcomposition_new() {
        let item = ModelComposition::new("test".to_string(), "test".to_string(), "test".to_string(), 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_modelweakening_new() {
        let item = ModelWeakening::new(Vec::new(), Vec::new(), 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_modelstrengthening_new() {
        let item = ModelStrengthening::new(Vec::new(), Vec::new(), 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_litmustransformation_new() {
        let item = LitmusTransformation::new("test".to_string(), "test".to_string(), "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_outcomeprediction_new() {
        let item = OutcomePrediction::new("test".to_string(), Vec::new(), 0.0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_behaviorcounter_new() {
        let item = BehaviorCounter::new(0, 0, 0);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_modelfeaturevector_new() {
        let item = ModelFeatureVector::new(Vec::new(), 0, "test".to_string());
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_diffpatch_new() {
        let item = DiffPatch::new(Vec::new(), false, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_modeldistance_new() {
        let item = ModelDistanceExt::new("test".to_string(), 0.0, false);
        let _ = format!("{:?}", item);
        let _ = format!("{}", item);
    }

    #[test]
    fn test_diff_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = diff_mean(&data);
        assert!((m - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = diff_variance(&data);
        assert!(v > 0.0);
    }

    #[test]
    fn test_diff_median() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let m = diff_median(&data);
        assert!((m - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_entropy() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let e = diff_entropy(&data);
        assert!(e > 0.0);
    }

    #[test]
    fn test_diff_std_dev() {
        let data = vec![10.0, 10.0, 10.0];
        let s = diff_std_dev(&data);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_diff_analysis() {
        let mut a = DiffAnalysis::new(3);
        a.set(0, 1, 0.5);
        a.set(1, 2, 0.3);
        assert!((a.get(0, 1) - 0.5).abs() < 1e-10);
        assert_eq!(a.size, 3);
    }

    #[test]
    fn test_diff_iterator() {
        let iter = DiffResultIterator::new(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let items: Vec<_> = iter.collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_diff_batch_processor() {
        let mut proc = DiffBatchProcessor::new(2);
        proc.process_batch(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(proc.processed, 4);
        assert_eq!(proc.results.len(), 2);
    }

    #[test]
    fn test_diff_histogram() {
        let hist = DiffHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(hist.num_bins(), 3);
        assert_eq!(hist.total_count, 5);
    }

    #[test]
    fn test_diff_graph() {
        let mut g = DiffGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 3.0);
        assert_eq!(g.edge_count, 3);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(3, 0));
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_diff_graph_shortest_path() {
        let mut g = DiffGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        let dist = g.shortest_path_dijkstra(0);
        assert!((dist[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_graph_topo_sort() {
        let mut g = DiffGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        let topo = g.topological_sort();
        assert!(topo.is_some());
    }

    #[test]
    fn test_diff_graph_components() {
        let mut g = DiffGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 1.0);
        let comps = g.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_diff_cache() {
        let mut cache = DiffCache::new(10);
        cache.insert(42, vec![1.0, 2.0]);
        assert!(cache.get(42).is_some());
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_diff_config() {
        let config = DiffConfig::default_config().with_verbose(true).with_max_iterations(500);
        assert!(config.verbose);
        assert_eq!(config.max_iterations, 500);
    }

    #[test]
    fn test_diff_report() {
        let mut report = DiffReport::new("Test Report");
        report.add_metric("accuracy", 0.95);
        report.add_warning("low sample size");
        assert_eq!(report.total_metrics(), 1);
        assert!(report.has_warnings());
        let text = report.render_text();
        assert!(text.contains("Test Report"));
    }

    #[test]
    fn test_diff_kmeans() {
        let data = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![10.0, 10.0], vec![10.1, 10.1]];
        let assignments = diff_kmeans(&data, 2, 100);
        assert_eq!(assignments.len(), 4);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
    }

    #[test]
    fn test_diff_pairwise_distances() {
        let points = vec![vec![0.0, 0.0], vec![3.0, 4.0]];
        let dists = diff_pairwise_distances(&points);
        assert!((dists[0][1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_harmmean() {
        let data = vec![1.0, 2.0, 4.0];
        let hm = diff_harmmean(&data);
        assert!(hm > 0.0 && hm < 4.0);
    }

    #[test]
    fn test_diff_geomean() {
        let data = vec![1.0, 2.0, 4.0];
        let gm = diff_geomean(&data);
        assert!(gm > 0.0 && gm < 4.0);
    }

    #[test]
    fn test_diff_sample_skewness() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = diff_sample_skewness(&data);
        assert!(s.abs() < 1.0);
    }

    #[test]
    fn test_diff_excess_kurtosis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = diff_excess_kurtosis(&data);
        let _ = k; // Just verify it computes
    }

    #[test]
    fn test_diff_gini() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        let g = diff_gini(&data);
        assert!(g.abs() < 0.01);
    }

    #[test]
    fn test_diff_percentile_at() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let p = diff_percentile_at(&data, 0.9);
        assert!(p >= 9.0);
    }

    #[test]
    fn test_diff_pca_2d() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let proj = diff_pca_2d(&data);
        assert_eq!(proj.len(), 2);
    }

    #[test]
    fn test_diff_analysis_normalize() {
        let mut a = DiffAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(0, 1, 3.0);
        a.normalize();
        assert!((a.total_sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_analysis_transpose() {
        let mut a = DiffAnalysis::new(2);
        a.set(0, 1, 5.0);
        let t = a.transpose();
        assert!((t.get(1, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_analysis_multiply() {
        let mut a = DiffAnalysis::new(2);
        a.set(0, 0, 1.0); a.set(1, 1, 1.0);
        let mut b = DiffAnalysis::new(2);
        b.set(0, 1, 2.0); b.set(1, 0, 3.0);
        let c = a.multiply(&b);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_analysis_frobenius() {
        let mut a = DiffAnalysis::new(2);
        a.set(0, 0, 3.0); a.set(1, 1, 4.0);
        assert!((a.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_analysis_symmetric() {
        let mut a = DiffAnalysis::new(2);
        a.set(0, 1, 1.0); a.set(1, 0, 1.0);
        assert!(a.is_symmetric());
    }

    #[test]
    fn test_diff_graph_dot() {
        let mut g = DiffGraph::new(2);
        g.add_edge(0, 1, 1.0);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_diff_histogram_render() {
        let hist = DiffHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 5);
        let ascii = hist.render_ascii(20);
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_diff_batch_reset() {
        let mut proc = DiffBatchProcessor::new(3);
        proc.process_batch(&[1.0, 2.0, 3.0]);
        assert!(proc.processed > 0);
        proc.reset();
        assert_eq!(proc.processed, 0);
    }

    #[test]
    fn test_diff_graph_remove_edge() {
        let mut g = DiffGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        assert_eq!(g.edge_count, 2);
        g.remove_edge(0, 1);
        assert_eq!(g.edge_count, 1);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_diff_dense_matrix_new() {
        let m = DiffDenseMatrix::new(3, 3);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
    }

    #[test]
    fn test_diff_dense_matrix_identity() {
        let m = DiffDenseMatrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_mul() {
        let a = DiffDenseMatrix::identity(2);
        let b = DiffDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a.mul_matrix(&b);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_transpose() {
        let a = DiffDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_det_2x2() {
        let m = DiffDenseMatrix::from_vec(2, 2, vec![3.0, 7.0, 1.0, -4.0]);
        let det = m.determinant_2x2();
        assert!((det - (-19.0)).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_det_3x3() {
        let m = DiffDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
        let det = m.determinant_3x3();
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_inverse_2x2() {
        let m = DiffDenseMatrix::from_vec(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        let inv = m.inverse_2x2().unwrap();
        let prod = m.mul_matrix(&inv);
        assert!((prod.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_power() {
        let m = DiffDenseMatrix::identity(3);
        let p = m.power(5);
        assert!((p.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_rank() {
        let m = DiffDenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        assert_eq!(m.rank(), 1);
    }

    #[test]
    fn test_diff_dense_matrix_solve() {
        let a = DiffDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 5.0, 3.0]);
        let x = a.solve(&[4.0, 7.0]).unwrap();
        assert!((x[0] - 5.0).abs() < 1e-8);
        assert!((x[1] - (-6.0)).abs() < 1e-8);
    }

    #[test]
    fn test_diff_dense_matrix_lu() {
        let a = DiffDenseMatrix::from_vec(2, 2, vec![4.0, 3.0, 6.0, 3.0]);
        let (l, u) = a.lu_decomposition();
        let prod = l.mul_matrix(&u);
        assert!((prod.get(0, 0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_eigenvalues() {
        let m = DiffDenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (e1, e2) = m.eigenvalues_2x2();
        assert!((e1 - 3.0).abs() < 1e-10);
        assert!((e2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_kronecker() {
        let a = DiffDenseMatrix::identity(2);
        let b = DiffDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let k = a.kronecker_product(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
    }

    #[test]
    fn test_diff_dense_matrix_hadamard() {
        let a = DiffDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = DiffDenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let h = a.hadamard_product(&b);
        assert!((h.get(0, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_interval() {
        let a = DiffInterval::new(1.0, 3.0);
        let b = DiffInterval::new(2.0, 5.0);
        assert!(a.overlaps(&b));
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
        let sum = a.add(&b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_interval_mul() {
        let a = DiffInterval::new(-2.0, 3.0);
        let b = DiffInterval::new(1.0, 4.0);
        let prod = a.mul(&b);
        assert!((prod.lo - (-8.0)).abs() < 1e-10);
        assert!((prod.hi - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_interval_hull() {
        let a = DiffInterval::new(1.0, 3.0);
        let b = DiffInterval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert!((h.lo - 1.0).abs() < 1e-10);
        assert!((h.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_state_machine() {
        let mut sm = DiffStateMachine::new();
        assert_eq!(*sm.state(), DiffState::Empty);
        assert!(sm.transition(DiffState::Loaded));
        assert_eq!(*sm.state(), DiffState::Loaded);
        assert_eq!(sm.transition_count, 1);
    }

    #[test]
    fn test_diff_state_machine_invalid() {
        let mut sm = DiffStateMachine::new();
        let last_state = DiffState::Reverted;
        assert!(!sm.can_transition(&last_state));
    }

    #[test]
    fn test_diff_state_machine_reset() {
        let mut sm = DiffStateMachine::new();
        sm.transition(DiffState::Loaded);
        sm.reset();
        assert_eq!(*sm.state(), DiffState::Empty);
        assert_eq!(sm.history_len(), 0);
    }

    #[test]
    fn test_diff_ring_buffer() {
        let mut rb = DiffRingBuffer::new(3);
        rb.push(1.0); rb.push(2.0); rb.push(3.0);
        assert!(rb.is_full());
        assert!((rb.average() - 2.0).abs() < 1e-10);
        rb.push(4.0);
        assert!((rb.oldest().unwrap() - 2.0).abs() < 1e-10);
        assert!((rb.latest().unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_ring_buffer_to_vec() {
        let mut rb = DiffRingBuffer::new(5);
        rb.push(10.0); rb.push(20.0); rb.push(30.0);
        let v = rb.to_vec();
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_disjoint_set() {
        let mut ds = DiffDisjointSet::new(5);
        assert_eq!(ds.num_components(), 5);
        ds.union(0, 1);
        ds.union(2, 3);
        assert_eq!(ds.num_components(), 3);
        assert!(ds.connected(0, 1));
        assert!(!ds.connected(0, 2));
    }

    #[test]
    fn test_diff_disjoint_set_components() {
        let mut ds = DiffDisjointSet::new(4);
        ds.union(0, 1); ds.union(2, 3);
        let comps = ds.components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_diff_sorted_list() {
        let mut sl = DiffSortedList::new();
        sl.insert(3.0); sl.insert(1.0); sl.insert(2.0);
        assert_eq!(sl.len(), 3);
        assert!((sl.min().unwrap() - 1.0).abs() < 1e-10);
        assert!((sl.max().unwrap() - 3.0).abs() < 1e-10);
        assert!((sl.median() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sorted_list_remove() {
        let mut sl = DiffSortedList::new();
        sl.insert(1.0); sl.insert(2.0); sl.insert(3.0);
        assert!(sl.remove(2.0));
        assert_eq!(sl.len(), 2);
        assert!(!sl.contains(2.0));
    }

    #[test]
    fn test_diff_ema() {
        let mut ema = DiffEma::new(0.5);
        ema.update(10.0);
        assert!((ema.current() - 10.0).abs() < 1e-10);
        ema.update(20.0);
        assert!((ema.current() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_bloom_filter() {
        let mut bf = DiffBloomFilter::new(1000, 3);
        bf.insert(42);
        bf.insert(100);
        assert!(bf.may_contain(42));
        assert!(bf.may_contain(100));
        assert_eq!(bf.count(), 2);
    }

    #[test]
    fn test_diff_trie() {
        let mut trie = DiffTrie::new();
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
    fn test_diff_dense_matrix_sym() {
        let m = DiffDenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric());
    }

    #[test]
    fn test_diff_dense_matrix_diag() {
        let m = DiffDenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
        assert!(m.is_diagonal());
    }

    #[test]
    fn test_diff_dense_matrix_upper_tri() {
        let m = DiffDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        assert!(m.is_upper_triangular());
    }

    #[test]
    fn test_diff_dense_matrix_outer() {
        let m = DiffDenseMatrix::outer_product(&[1.0, 2.0], &[3.0, 4.0]);
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dense_matrix_submatrix() {
        let m = DiffDenseMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let sub = m.submatrix(0, 0, 2, 2);
        assert_eq!(sub.rows, 2);
        assert!((sub.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_priority_queue() {
        let mut pq = DiffPriorityQueue::new();
        pq.push(3.0, 1); pq.push(1.0, 2); pq.push(2.0, 3);
        assert_eq!(pq.pop().unwrap().1, 2);
        assert_eq!(pq.pop().unwrap().1, 3);
        assert_eq!(pq.pop().unwrap().1, 1);
    }

    #[test]
    fn test_diff_accumulator() {
        let mut acc = DiffAccumulator::new();
        for i in 1..=10 { acc.add(i as f64); }
        assert!((acc.mean() - 5.5).abs() < 1e-10);
        assert_eq!(acc.count(), 10);
        assert!((acc.min() - 1.0).abs() < 1e-10);
        assert!((acc.max() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_accumulator_merge() {
        let mut a = DiffAccumulator::new();
        a.add(1.0); a.add(2.0);
        let mut b = DiffAccumulator::new();
        b.add(3.0); b.add(4.0);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert!((a.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sparse_matrix() {
        let mut m = DiffSparseMatrix::new(3, 3);
        m.insert(0, 1, 2.0); m.insert(1, 2, 3.0);
        assert_eq!(m.nnz(), 2);
        assert!((m.get(0, 1) - 2.0).abs() < 1e-10);
        assert!((m.get(0, 0)).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sparse_mul_vec() {
        let mut m = DiffSparseMatrix::new(2, 2);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0);
        let result = m.mul_vec(&[3.0, 4.0]);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sparse_transpose() {
        let mut m = DiffSparseMatrix::new(2, 3);
        m.insert(0, 2, 5.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3); assert_eq!(t.cols, 2);
        assert!((t.get(2, 0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_polynomial_eval() {
        let p = DiffPolynomial::new(vec![1.0, 2.0, 3.0]);
        assert!((p.evaluate(2.0) - 17.0).abs() < 1e-10);
        assert!((p.evaluate_horner(2.0) - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_polynomial_add() {
        let a = DiffPolynomial::new(vec![1.0, 2.0]);
        let b = DiffPolynomial::new(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert!((c.evaluate(1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_polynomial_mul() {
        let a = DiffPolynomial::new(vec![1.0, 1.0]);
        let b = DiffPolynomial::new(vec![1.0, 1.0]);
        let c = a.mul(&b);
        assert_eq!(c.degree(), 2);
        assert!((c.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_polynomial_deriv() {
        let p = DiffPolynomial::new(vec![1.0, 2.0, 3.0]);
        let dp = p.derivative();
        assert!((dp.evaluate(0.0) - 2.0).abs() < 1e-10);
        assert!((dp.evaluate(1.0) - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_polynomial_integral() {
        let p = DiffPolynomial::new(vec![2.0, 3.0]);
        let ip = p.integral(0.0);
        assert!((ip.evaluate(1.0) - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_diff_polynomial_roots() {
        let p = DiffPolynomial::new(vec![-6.0, 1.0, 1.0]);
        let roots = p.roots_quadratic();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_diff_polynomial_newton() {
        let p = DiffPolynomial::new(vec![-2.0, 0.0, 1.0]);
        let root = p.newton_root(1.0, 100, 1e-10);
        assert!(root.is_some());
        assert!((root.unwrap() - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_diff_polynomial_compose() {
        let p = DiffPolynomial::new(vec![0.0, 0.0, 1.0]);
        let q = DiffPolynomial::new(vec![1.0, 1.0]);
        let r = p.compose(&q);
        assert!((r.evaluate(2.0) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_rng() {
        let mut rng = DiffRng::new(42);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_diff_rng_gaussian() {
        let mut rng = DiffRng::new(123);
        let mut sum = 0.0;
        for _ in 0..1000 { sum += rng.next_gaussian(); }
        let mean = sum / 1000.0;
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_diff_timer() {
        let mut timer = DiffTimer::new("test");
        timer.record(100); timer.record(200); timer.record(300);
        assert_eq!(timer.count(), 3);
        assert_eq!(timer.total_ns(), 600);
        assert!((timer.average_ns() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_bitvector() {
        let mut bv = DiffBitVector::new(100);
        bv.set(5); bv.set(42); bv.set(99);
        assert!(bv.get(5));
        assert!(bv.get(42));
        assert!(!bv.get(50));
        assert_eq!(bv.count_ones(), 3);
    }

    #[test]
    fn test_diff_bitvector_ops() {
        let mut a = DiffBitVector::new(64);
        a.set(0); a.set(10); a.set(20);
        let mut b = DiffBitVector::new(64);
        b.set(10); b.set(20); b.set(30);
        let c = a.and(&b);
        assert_eq!(c.count_ones(), 2);
        let d = a.or(&b);
        assert_eq!(d.count_ones(), 4);
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_diff_bitvector_jaccard() {
        let mut a = DiffBitVector::new(10);
        a.set(0); a.set(1); a.set(2);
        let mut b = DiffBitVector::new(10);
        b.set(1); b.set(2); b.set(3);
        let j = a.jaccard(&b);
        assert!((j - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_diff_priority_queue_empty() {
        let mut pq = DiffPriorityQueue::new();
        assert!(pq.is_empty());
        assert!(pq.pop().is_none());
    }

    #[test]
    fn test_diff_sparse_add() {
        let mut a = DiffSparseMatrix::new(2, 2);
        a.insert(0, 0, 1.0);
        let mut b = DiffSparseMatrix::new(2, 2);
        b.insert(0, 0, 2.0); b.insert(1, 1, 3.0);
        let c = a.add(&b);
        assert!((c.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((c.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_rng_shuffle() {
        let mut rng = DiffRng::new(99);
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        rng.shuffle(&mut data);
        assert_eq!(data.len(), 5);
        let sum: f64 = data.iter().sum();
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_polynomial_display() {
        let p = DiffPolynomial::new(vec![1.0, 2.0, 3.0]);
        let s = format!("{}", p);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_diff_polynomial_monomial() {
        let m = DiffPolynomial::monomial(3, 5.0);
        assert_eq!(m.degree(), 3);
        assert!((m.evaluate(2.0) - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_timer_percentiles() {
        let mut timer = DiffTimer::new("perf");
        for i in 1..=100 { timer.record(i); }
        assert_eq!(timer.p50_ns(), 50);
        assert!(timer.p95_ns() >= 90);
    }

    #[test]
    fn test_diff_accumulator_cv() {
        let mut acc = DiffAccumulator::new();
        acc.add(10.0); acc.add(10.0); acc.add(10.0);
        assert!(acc.coefficient_of_variation().abs() < 1e-10);
    }

    #[test]
    fn test_diff_sparse_diagonal() {
        let mut m = DiffSparseMatrix::new(3, 3);
        m.insert(0, 0, 1.0); m.insert(1, 1, 2.0); m.insert(2, 2, 3.0);
        assert!((m.trace() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_lru_cache() {
        let mut cache = DiffLruCache::new(2);
        cache.put(1, vec![1.0]);
        cache.put(2, vec![2.0]);
        assert!(cache.get(1).is_some());
        cache.put(3, vec![3.0]);
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_diff_lru_hit_rate() {
        let mut cache = DiffLruCache::new(10);
        cache.put(1, vec![1.0]);
        cache.get(1);
        cache.get(2);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_diff_graph_coloring() {
        let mut gc = DiffGraphColoring::new(4);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(2, 3); gc.add_edge(3, 0);
        let colors = gc.greedy_color();
        assert!(gc.is_valid_coloring());
        assert!(colors <= 3);
    }

    #[test]
    fn test_diff_graph_coloring_complete() {
        let mut gc = DiffGraphColoring::new(3);
        gc.add_edge(0, 1); gc.add_edge(1, 2); gc.add_edge(0, 2);
        let colors = gc.greedy_color();
        assert_eq!(colors, 3);
        assert!(gc.is_valid_coloring());
    }

    #[test]
    fn test_diff_topk() {
        let mut tk = DiffTopK::new(3);
        tk.insert(5.0, "e"); tk.insert(3.0, "c"); tk.insert(1.0, "a");
        tk.insert(4.0, "d"); tk.insert(2.0, "b");
        assert_eq!(tk.len(), 3);
        assert!((tk.max_score().unwrap() - 5.0).abs() < 1e-10);
        assert!((tk.min_score().unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sliding_window() {
        let mut sw = DiffSlidingWindow::new(3);
        sw.push(1.0); sw.push(2.0); sw.push(3.0);
        assert!((sw.mean() - 2.0).abs() < 1e-10);
        sw.push(4.0);
        assert!((sw.mean() - 3.0).abs() < 1e-10);
        assert_eq!(sw.len(), 3);
    }

    #[test]
    fn test_diff_sliding_window_trend() {
        let mut sw = DiffSlidingWindow::new(10);
        for i in 0..5 { sw.push(i as f64); }
        assert!(sw.trend() > 0.0);
    }

    #[test]
    fn test_diff_confusion_matrix() {
        let actual = vec![true, true, false, false, true];
        let predicted = vec![true, false, false, true, true];
        let cm = DiffConfusionMatrix::from_predictions(&actual, &predicted);
        assert_eq!(cm.true_positive, 2);
        assert_eq!(cm.false_positive, 1);
        assert_eq!(cm.true_negative, 1);
        assert_eq!(cm.false_negative, 1);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_diff_confusion_f1() {
        let cm = DiffConfusionMatrix { true_positive: 80, false_positive: 20, true_negative: 70, false_negative: 30 };
        assert!((cm.precision() - 0.8).abs() < 1e-10);
        let f1 = cm.f1_score();
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_diff_cosine_similarity() {
        let s = diff_cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-10);
        let s2 = diff_cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_euclidean_distance() {
        let d = diff_euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sigmoid() {
        let s = diff_sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-10);
        let s2 = diff_sigmoid(100.0);
        assert!((s2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_diff_softmax() {
        let sm = diff_softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(sm[2] > sm[1] && sm[1] > sm[0]);
    }

    #[test]
    fn test_diff_kl_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let kl = diff_kl_divergence(&p, &q);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_diff_normalize() {
        let v = diff_normalize(&[3.0, 4.0]);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_lerp() {
        assert!((diff_lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-10);
        assert!((diff_lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((diff_lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_clamp() {
        assert!((diff_clamp(5.0, 0.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((diff_clamp(-5.0, 0.0, 10.0) - 0.0).abs() < 1e-10);
        assert!((diff_clamp(15.0, 0.0, 10.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_cross_product() {
        let c = diff_cross_product(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!((c[0]).abs() < 1e-10);
        assert!((c[1]).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dot_product() {
        let d = diff_dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_js_divergence() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let js = diff_js_divergence(&p, &q);
        assert!(js.abs() < 1e-10);
    }

    #[test]
    fn test_diff_hellinger() {
        let p = vec![0.5, 0.5];
        let q = vec![0.5, 0.5];
        let h = diff_hellinger_distance(&p, &q);
        assert!(h.abs() < 1e-10);
    }

    #[test]
    fn test_diff_logsumexp() {
        let lse = diff_logsumexp(&[1.0, 2.0, 3.0]);
        assert!(lse > 3.0);
    }

    #[test]
    fn test_diff_feature_scaler() {
        let mut scaler = DiffFeatureScaler::new();
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        scaler.fit(&data);
        let normalized = scaler.normalize(&[2.0, 20.0]);
        assert!((normalized[0] - 0.5).abs() < 1e-10);
        assert!((normalized[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_diff_feature_scaler_inverse() {
        let mut scaler = DiffFeatureScaler::new();
        let data = vec![vec![0.0, 0.0], vec![10.0, 100.0]];
        scaler.fit(&data);
        let normed = scaler.normalize(&[5.0, 50.0]);
        let inv = scaler.inverse_normalize(&normed);
        assert!((inv[0] - 5.0).abs() < 1e-10);
        assert!((inv[1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_linear_regression() {
        let mut lr = DiffLinearRegression::new();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        lr.fit(&x, &y);
        assert!((lr.slope - 2.0).abs() < 1e-10);
        assert!(lr.intercept.abs() < 1e-10);
        assert!((lr.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_linear_regression_predict() {
        let mut lr = DiffLinearRegression::new();
        lr.fit(&[0.0, 1.0, 2.0], &[1.0, 3.0, 5.0]);
        assert!((lr.predict(3.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_weighted_graph() {
        let mut g = DiffWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        assert_eq!(g.num_edges, 3);
        let dists = g.dijkstra(0);
        assert!((dists[3] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_weighted_graph_mst() {
        let mut g = DiffWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 2.0); g.add_edge(2, 3, 3.0);
        g.add_edge(0, 3, 10.0);
        let mst = g.min_spanning_tree_weight();
        assert!((mst - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = diff_moving_average(&data, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_cumsum() {
        let cs = diff_cumsum(&[1.0, 2.0, 3.0, 4.0]);
        assert!((cs[3] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_diff() {
        let d = diff_diff(&[1.0, 3.0, 6.0, 10.0]);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_autocorrelation() {
        let ac = diff_autocorrelation(&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0], 0);
        assert!((ac - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_dft_magnitude() {
        let mags = diff_dft_magnitude(&[1.0, 0.0, -1.0, 0.0]);
        assert!(!mags.is_empty());
    }

    #[test]
    fn test_diff_integrate_trapezoid() {
        let area = diff_integrate_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_convolve() {
        let c = diff_convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_weighted_graph_clustering() {
        let mut g = DiffWeightedGraph::new(4);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        let cc = g.clustering_coefficient(0);
        assert!((cc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_histogram_cumulative() {
        let h = DiffHistogramExt::from_data(&[1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);
        let cum = h.cumulative();
        assert_eq!(*cum.last().unwrap(), 6);
    }

    #[test]
    fn test_diff_histogram_entropy() {
        let h = DiffHistogramExt::from_data(&[1.0, 2.0, 3.0, 4.0], 4);
        let ent = h.entropy();
        assert!(ent > 0.0);
    }

    #[test]
    fn test_diff_aabb() {
        let bb = DiffAABB::new(0.0, 0.0, 10.0, 10.0);
        assert!(bb.contains(5.0, 5.0));
        assert!(!bb.contains(11.0, 5.0));
        assert!((bb.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_aabb_intersects() {
        let a = DiffAABB::new(0.0, 0.0, 10.0, 10.0);
        let b = DiffAABB::new(5.0, 5.0, 15.0, 15.0);
        let c = DiffAABB::new(20.0, 20.0, 30.0, 30.0);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_diff_quadtree() {
        let bb = DiffAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = DiffQuadTree::new(bb, 4, 8);
        for i in 0..20 {
            qt.insert(DiffPoint2D { x: i as f64 * 5.0, y: i as f64 * 5.0, data: i as f64 });
        }
        assert_eq!(qt.count(), 20);
    }

    #[test]
    fn test_diff_quadtree_query() {
        let bb = DiffAABB::new(0.0, 0.0, 100.0, 100.0);
        let mut qt = DiffQuadTree::new(bb, 2, 8);
        qt.insert(DiffPoint2D { x: 10.0, y: 10.0, data: 1.0 });
        qt.insert(DiffPoint2D { x: 90.0, y: 90.0, data: 2.0 });
        let range = DiffAABB::new(0.0, 0.0, 50.0, 50.0);
        let found = qt.query_range(&range);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_diff_mat_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = diff_mat_mul(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 1e-10);
        assert!((c[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_transpose() {
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = diff_transpose(&a);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[2][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_frobenius_norm() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let n = diff_frobenius_norm(&a);
        assert!((n - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_diff_trace() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!((diff_trace(&a) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_identity() {
        let id = diff_identity(3);
        assert!((id[0][0] - 1.0).abs() < 1e-10);
        assert!((id[0][1]).abs() < 1e-10);
        assert!((id[2][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_power_iteration() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, _evec) = diff_power_iteration(&a, 100);
        assert!((eval - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_diff_running_stats() {
        let mut s = DiffRunningStats::new();
        for &v in &[1.0, 2.0, 3.0, 4.0, 5.0] { s.push(v); }
        assert!((s.mean - 3.0).abs() < 1e-10);
        assert!((s.min_val - 1.0).abs() < 1e-10);
        assert!((s.max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_running_stats_merge() {
        let mut a = DiffRunningStats::new();
        let mut b = DiffRunningStats::new();
        for &v in &[1.0, 2.0, 3.0] { a.push(v); }
        for &v in &[4.0, 5.0, 6.0] { b.push(v); }
        a.merge(&b);
        assert_eq!(a.count, 6);
        assert!((a.mean - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_diff_running_stats_cv() {
        let mut s = DiffRunningStats::new();
        for &v in &[10.0, 10.0, 10.0] { s.push(v); }
        assert!(s.coefficient_of_variation() < 1e-10);
    }

    #[test]
    fn test_diff_iqr() {
        let iqr = diff_iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_diff_outliers() {
        let outliers = diff_outliers(&[1.0, 2.0, 3.0, 4.0, 5.0, 100.0]);
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_diff_zscore() {
        let z = diff_zscore(&[10.0, 20.0, 30.0]);
        assert!((z[1]).abs() < 1e-10); // middle value should be ~0
    }

    #[test]
    fn test_diff_rank() {
        let r = diff_rank(&[30.0, 10.0, 20.0]);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_spearman() {
        let rho = diff_spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_sample_skewness_symmetric() {
        let s = diff_sample_skewness(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_diff_covariance_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let cov = diff_covariance_matrix(&data);
        assert_eq!(cov.len(), 2);
        assert!(cov[0][0] > 0.0);
    }

    #[test]
    fn test_diff_correlation_matrix() {
        let data = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let corr = diff_correlation_matrix(&data);
        assert!((corr[0][1] - 1.0).abs() < 1e-10); // perfect correlation
    }

}

//! Core leakage contract types.
//!
//! A leakage contract for a function *f* has the form:
//!
//! ```text
//! Contract_f = (τ_f : CacheState → CacheState,  B_f : CacheState → ℝ≥0)
//! ```

use std::collections::BTreeMap;
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use shared_types::{CacheSet, FunctionId, SecurityLevel, VirtualAddress, CacheGeometry};

// ---------------------------------------------------------------------------
// Abstract cache state
// ---------------------------------------------------------------------------

/// State of a single cache line inside a set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CacheLineState {
    /// The line is known to hold a specific tag (exact).
    Known { tag: u64 },
    /// The line may hold any tag (fully unknown).
    Unknown,
    /// The line is empty / invalid.
    Empty,
    /// The line holds one of several possible tags.
    OneOf(Vec<u64>),
}

impl CacheLineState {
    /// Returns `true` if the line state is fully determined.
    pub fn is_exact(&self) -> bool {
        matches!(self, CacheLineState::Known { .. })
    }

    /// Join (least upper bound) of two cache-line states.
    pub fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (CacheLineState::Known { tag: a }, CacheLineState::Known { tag: b }) if a == b => {
                CacheLineState::Known { tag: *a }
            }
            (CacheLineState::Empty, CacheLineState::Empty) => CacheLineState::Empty,
            (CacheLineState::Known { tag: a }, CacheLineState::Known { tag: b }) => {
                CacheLineState::OneOf(vec![*a, *b])
            }
            (CacheLineState::OneOf(tags), CacheLineState::Known { tag }) |
            (CacheLineState::Known { tag }, CacheLineState::OneOf(tags)) => {
                let mut merged = tags.clone();
                if !merged.contains(tag) {
                    merged.push(*tag);
                }
                merged.sort_unstable();
                CacheLineState::OneOf(merged)
            }
            (CacheLineState::OneOf(a), CacheLineState::OneOf(b)) => {
                let mut merged = a.clone();
                for t in b {
                    if !merged.contains(t) {
                        merged.push(*t);
                    }
                }
                merged.sort_unstable();
                CacheLineState::OneOf(merged)
            }
            _ => CacheLineState::Unknown,
        }
    }

    /// Meet (greatest lower bound) of two cache-line states.
    pub fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (CacheLineState::Unknown, x) | (x, CacheLineState::Unknown) => x.clone(),
            (CacheLineState::Known { tag: a }, CacheLineState::Known { tag: b }) if a == b => {
                CacheLineState::Known { tag: *a }
            }
            (CacheLineState::Empty, CacheLineState::Empty) => CacheLineState::Empty,
            (CacheLineState::OneOf(tags), CacheLineState::Known { tag }) |
            (CacheLineState::Known { tag }, CacheLineState::OneOf(tags)) => {
                if tags.contains(tag) {
                    CacheLineState::Known { tag: *tag }
                } else {
                    CacheLineState::Empty
                }
            }
            (CacheLineState::OneOf(a), CacheLineState::OneOf(b)) => {
                let inter: Vec<u64> = a.iter().copied().filter(|t| b.contains(t)).collect();
                match inter.len() {
                    0 => CacheLineState::Empty,
                    1 => CacheLineState::Known { tag: inter[0] },
                    _ => CacheLineState::OneOf(inter),
                }
            }
            _ => CacheLineState::Empty,
        }
    }

    /// Information content: how many bits of uncertainty this line carries.
    pub fn uncertainty_bits(&self) -> f64 {
        match self {
            CacheLineState::Known { .. } | CacheLineState::Empty => 0.0,
            CacheLineState::Unknown => 64.0,
            CacheLineState::OneOf(tags) => (tags.len() as f64).log2(),
        }
    }
}

impl fmt::Display for CacheLineState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheLineState::Known { tag } => write!(f, "Known(0x{:x})", tag),
            CacheLineState::Unknown => write!(f, "⊤"),
            CacheLineState::Empty => write!(f, "⊥"),
            CacheLineState::OneOf(tags) => {
                let strs: Vec<String> = tags.iter().map(|t| format!("0x{:x}", t)).collect();
                write!(f, "{{{}}}", strs.join(", "))
            }
        }
    }
}

/// Abstract state of a single cache set (multiple ways).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheSetState {
    /// Per-way abstract line states, index 0 = MRU.
    pub ways: Vec<CacheLineState>,
    /// Whether this set has been touched during analysis.
    pub touched: bool,
    /// The number of *distinct* accesses observed.
    pub access_count: u64,
}

impl CacheSetState {
    /// Create a new set state with the given associativity, all lines empty.
    pub fn new_empty(associativity: usize) -> Self {
        Self {
            ways: vec![CacheLineState::Empty; associativity],
            touched: false,
            access_count: 0,
        }
    }

    /// Create a fully-unknown set state.
    pub fn new_unknown(associativity: usize) -> Self {
        Self {
            ways: vec![CacheLineState::Unknown; associativity],
            touched: false,
            access_count: 0,
        }
    }

    /// Point-wise join.
    pub fn join(&self, other: &Self) -> Self {
        let len = self.ways.len().max(other.ways.len());
        let mut ways = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.ways.get(i).cloned().unwrap_or(CacheLineState::Empty);
            let b = other.ways.get(i).cloned().unwrap_or(CacheLineState::Empty);
            ways.push(a.join(&b));
        }
        Self {
            ways,
            touched: self.touched || other.touched,
            access_count: self.access_count.max(other.access_count),
        }
    }

    /// Point-wise meet.
    pub fn meet(&self, other: &Self) -> Self {
        let len = self.ways.len().min(other.ways.len());
        let mut ways = Vec::with_capacity(len);
        for i in 0..len {
            ways.push(self.ways[i].meet(&other.ways[i]));
        }
        Self {
            ways,
            touched: self.touched && other.touched,
            access_count: self.access_count.min(other.access_count),
        }
    }

    /// Total uncertainty in bits for this set.
    pub fn uncertainty_bits(&self) -> f64 {
        self.ways.iter().map(|w| w.uncertainty_bits()).sum()
    }

    /// Number of ways that are exactly known.
    pub fn known_ways(&self) -> usize {
        self.ways.iter().filter(|w| w.is_exact()).count()
    }

    /// Associativity.
    pub fn associativity(&self) -> usize {
        self.ways.len()
    }

    /// Simulate an LRU access to this set with the given tag.
    pub fn lru_access(&mut self, tag: u64) {
        // Check if tag is already present
        let mut hit_idx = None;
        for (i, w) in self.ways.iter().enumerate() {
            if let CacheLineState::Known { tag: t } = w {
                if *t == tag {
                    hit_idx = Some(i);
                    break;
                }
            }
        }
        match hit_idx {
            Some(idx) => {
                // Move to MRU position
                let line = self.ways.remove(idx);
                self.ways.insert(0, line);
            }
            None => {
                // Cold miss: evict LRU, insert at MRU
                if !self.ways.is_empty() {
                    self.ways.pop();
                }
                self.ways.insert(0, CacheLineState::Known { tag });
            }
        }
        self.touched = true;
        self.access_count += 1;
    }
}

impl fmt::Display for CacheSetState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, w) in self.ways.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", w)?;
        }
        write!(f, "] (accesses={})", self.access_count)
    }
}

/// Abstract cache state: maps each cache set index to its abstract set state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractCacheState {
    /// Per-set abstract states, keyed by set index.
    pub sets: BTreeMap<u32, CacheSetState>,
    /// Default associativity for sets not yet materialised.
    pub default_associativity: usize,
    /// Description of the cache geometry this state refers to.
    pub geometry: CacheGeometryInfo,
}

/// Lightweight description of cache geometry (avoids pulling in the full
/// `CacheGeometry` from `shared_types` for serialisation friendliness).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheGeometryInfo {
    pub num_sets: u32,
    pub associativity: u32,
    pub line_size_bits: u32,
}

impl CacheGeometryInfo {
    pub fn from_shared(g: &CacheGeometry) -> Self {
        Self {
            num_sets: g.num_sets,
            associativity: g.num_ways,
            line_size_bits: g.line_size_bits,
        }
    }
}

impl AbstractCacheState {
    /// New state with every set empty.
    pub fn new_empty(geometry: CacheGeometryInfo) -> Self {
        Self {
            sets: BTreeMap::new(),
            default_associativity: geometry.associativity as usize,
            geometry,
        }
    }

    /// New fully-unknown state (all sets materialised as Unknown).
    pub fn new_unknown(geometry: CacheGeometryInfo) -> Self {
        let assoc = geometry.associativity as usize;
        let mut sets = BTreeMap::new();
        for s in 0..geometry.num_sets {
            sets.insert(s, CacheSetState::new_unknown(assoc));
        }
        Self {
            sets,
            default_associativity: assoc,
            geometry,
        }
    }

    /// Get (or lazily create) the state for a given set.
    pub fn get_set(&self, set_idx: u32) -> CacheSetState {
        self.sets
            .get(&set_idx)
            .cloned()
            .unwrap_or_else(|| CacheSetState::new_empty(self.default_associativity))
    }

    /// Mutably get (or lazily create) the state for a given set.
    pub fn get_set_mut(&mut self, set_idx: u32) -> &mut CacheSetState {
        let assoc = self.default_associativity;
        self.sets
            .entry(set_idx)
            .or_insert_with(|| CacheSetState::new_empty(assoc))
    }

    /// Point-wise join over all sets.
    pub fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (&idx, other_set) in &other.sets {
            let self_set = self.get_set(idx);
            result.sets.insert(idx, self_set.join(other_set));
        }
        result
    }

    /// Point-wise meet over all sets.
    pub fn meet(&self, other: &Self) -> Self {
        let mut result = Self::new_empty(self.geometry.clone());
        for (&idx, self_set) in &self.sets {
            if let Some(other_set) = other.sets.get(&idx) {
                result.sets.insert(idx, self_set.meet(other_set));
            }
        }
        result
    }

    /// Total uncertainty in bits across all materialised sets.
    pub fn total_uncertainty_bits(&self) -> f64 {
        self.sets.values().map(|s| s.uncertainty_bits()).sum()
    }

    /// Number of sets that have been touched.
    pub fn touched_sets(&self) -> usize {
        self.sets.values().filter(|s| s.touched).count()
    }

    /// Number of materialised sets.
    pub fn materialised_sets(&self) -> usize {
        self.sets.len()
    }

    /// Simulate an LRU access to the given virtual address.
    pub fn lru_access(&mut self, addr: &VirtualAddress) {
        let line_bits = self.geometry.line_size_bits;
        let set_bits = (self.geometry.num_sets as f64).log2() as u32;
        let set_idx = ((addr.0 >> line_bits) & ((1u64 << set_bits) - 1)) as u32;
        let tag = addr.0 >> (line_bits + set_bits);
        self.get_set_mut(set_idx).lru_access(tag);
    }

    /// Check whether `self` is a sound over-approximation of `concrete`.
    /// That is, every Known line in `self` is also Known in `concrete` at the
    /// same position.
    pub fn over_approximates(&self, concrete: &Self) -> bool {
        for (&idx, c_set) in &concrete.sets {
            let a_set = self.get_set(idx);
            for (i, c_line) in c_set.ways.iter().enumerate() {
                if let Some(a_line) = a_set.ways.get(i) {
                    match (a_line, c_line) {
                        (CacheLineState::Known { tag: at }, CacheLineState::Known { tag: ct }) => {
                            if at != ct {
                                return false;
                            }
                        }
                        (CacheLineState::Empty, CacheLineState::Known { .. }) => return false,
                        _ => { /* Unknown or OneOf over-approximates anything */ }
                    }
                }
            }
        }
        true
    }
}

impl fmt::Display for AbstractCacheState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AbstractCacheState ({} sets materialised):", self.sets.len())?;
        for (&idx, set) in &self.sets {
            writeln!(f, "  set {:>4}: {}", idx, set)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Cache transformer
// ---------------------------------------------------------------------------

/// Describes how a function transforms the abstract cache state.
///
/// ```text
/// τ_f : CacheState → CacheState
/// ```
///
/// Internally represented as a set of per-set transform entries so that the
/// transformer can be stored, serialised, and composed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheTransformer {
    /// Per-set transform rules: set_index → output state.
    /// Sets not present in this map are unmodified.
    pub set_transforms: BTreeMap<u32, SetTransformRule>,
    /// Whether the transformer is identity (touches nothing).
    pub is_identity: bool,
    /// Sets that are *read* (input dependence).
    pub reads: Vec<u32>,
    /// Sets that are *written* (output dependence).
    pub writes: Vec<u32>,
    /// A human-readable summary.
    pub description: String,
}

/// Rule for transforming a single cache set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SetTransformRule {
    /// The set state is replaced entirely.
    Replace(CacheSetState),
    /// An LRU access with the given tag is applied.
    LruAccess { tag: u64 },
    /// The set is joined with an additional state (weak update).
    WeakUpdate(CacheSetState),
    /// A sequence of LRU accesses.
    AccessSequence(Vec<u64>),
    /// The set is cleared.
    Clear,
    /// Identity – explicitly marks a set as untouched.
    Identity,
}

impl CacheTransformer {
    /// Identity transformer – leaves the cache state unmodified.
    pub fn identity() -> Self {
        Self {
            set_transforms: BTreeMap::new(),
            is_identity: true,
            reads: Vec::new(),
            writes: Vec::new(),
            description: "identity".into(),
        }
    }

    /// Create a transformer from a set of per-set rules.
    pub fn from_rules(rules: BTreeMap<u32, SetTransformRule>, description: impl Into<String>) -> Self {
        let is_identity = rules.values().all(|r| matches!(r, SetTransformRule::Identity));
        let writes: Vec<u32> = rules
            .iter()
            .filter(|(_, r)| !matches!(r, SetTransformRule::Identity))
            .map(|(&idx, _)| idx)
            .collect();
        let reads = writes.clone(); // conservative: assume reads ⊇ writes
        Self {
            set_transforms: rules,
            is_identity,
            reads,
            writes,
            description: description.into(),
        }
    }

    /// Apply this transformer to an abstract cache state.
    pub fn apply(&self, state: &AbstractCacheState) -> AbstractCacheState {
        if self.is_identity {
            return state.clone();
        }
        let mut result = state.clone();
        for (&set_idx, rule) in &self.set_transforms {
            match rule {
                SetTransformRule::Replace(new_state) => {
                    result.sets.insert(set_idx, new_state.clone());
                }
                SetTransformRule::LruAccess { tag } => {
                    result.get_set_mut(set_idx).lru_access(*tag);
                }
                SetTransformRule::WeakUpdate(update) => {
                    let current = result.get_set(set_idx);
                    result.sets.insert(set_idx, current.join(update));
                }
                SetTransformRule::AccessSequence(tags) => {
                    for tag in tags {
                        result.get_set_mut(set_idx).lru_access(*tag);
                    }
                }
                SetTransformRule::Clear => {
                    let assoc = result.default_associativity;
                    result.sets.insert(set_idx, CacheSetState::new_empty(assoc));
                }
                SetTransformRule::Identity => {}
            }
        }
        result
    }

    /// Compose two transformers: `self` followed by `other`.
    ///
    /// ```text
    /// (τ_g ∘ τ_f)(s) = τ_g(τ_f(s))
    /// ```
    pub fn compose(&self, other: &CacheTransformer) -> CacheTransformer {
        if self.is_identity {
            return other.clone();
        }
        if other.is_identity {
            return self.clone();
        }

        let mut combined = self.set_transforms.clone();
        for (&idx, rule) in &other.set_transforms {
            match rule {
                SetTransformRule::Identity => {}
                _ => {
                    combined.insert(idx, rule.clone());
                }
            }
        }

        let writes: Vec<u32> = combined
            .iter()
            .filter(|(_, r)| !matches!(r, SetTransformRule::Identity))
            .map(|(&idx, _)| idx)
            .collect();
        let mut reads = self.reads.clone();
        for r in &other.reads {
            if !reads.contains(r) {
                reads.push(*r);
            }
        }
        reads.sort_unstable();

        CacheTransformer {
            is_identity: combined.values().all(|r| matches!(r, SetTransformRule::Identity)),
            set_transforms: combined,
            reads,
            writes,
            description: format!("{}; {}", self.description, other.description),
        }
    }

    /// Number of sets modified.
    pub fn modified_sets(&self) -> usize {
        self.writes.len()
    }

    /// Whether this transformer touches a specific set.
    pub fn touches_set(&self, set_idx: u32) -> bool {
        self.writes.contains(&set_idx)
    }

    /// The set of cache sets that are both read and written.
    pub fn read_write_sets(&self) -> Vec<u32> {
        self.reads
            .iter()
            .filter(|r| self.writes.contains(r))
            .copied()
            .collect()
    }
}

impl fmt::Display for CacheTransformer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity {
            return write!(f, "τ = id");
        }
        writeln!(f, "τ ({} set rules):", self.set_transforms.len())?;
        for (&idx, rule) in &self.set_transforms {
            let rule_str = match rule {
                SetTransformRule::Replace(_) => "replace",
                SetTransformRule::LruAccess { tag } => {
                    return writeln!(f, "  set {:>4}: lru_access(0x{:x})", idx, tag);
                }
                SetTransformRule::WeakUpdate(_) => "weak_update",
                SetTransformRule::AccessSequence(tags) => {
                    return writeln!(f, "  set {:>4}: access_seq(len={})", idx, tags.len());
                }
                SetTransformRule::Clear => "clear",
                SetTransformRule::Identity => "id",
            };
            writeln!(f, "  set {:>4}: {}", idx, rule_str)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Leakage bound
// ---------------------------------------------------------------------------

/// A leakage bound: maps the *initial* abstract cache state to a worst-case
/// information leakage in bits (non-negative real).
///
/// ```text
/// B_f : CacheState → ℝ≥0
/// ```
///
/// Internally this is stored as either a constant bound, a per-set bound map,
/// or a full closure-like representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageBound {
    /// Kind of bound representation.
    pub kind: LeakageBoundKind,
    /// An upper bound that is always valid regardless of input state.
    pub worst_case_bits: f64,
    /// Optional per-set breakdown of the leakage.
    pub per_set_leakage: BTreeMap<u32, f64>,
    /// Whether this bound was proved tight (exact) or is an over-approximation.
    pub is_tight: bool,
    /// Description of how the bound was derived.
    pub derivation: String,
}

/// Kind of leakage bound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LeakageBoundKind {
    /// Constant bound: `B(s) = c` for all `s`.
    Constant(f64),
    /// Per-set linear bound: `B(s) = Σ_i b_i(s_i)`.
    PerSet(BTreeMap<u32, f64>),
    /// State-dependent bound described symbolically.
    Symbolic { expression: String },
    /// Bound derived from composition.
    Composed { components: Vec<f64>, operation: CompositionOp },
}

/// How a composed bound was computed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompositionOp {
    Sum,
    Max,
    SumWithTransform,
    LoopUnroll { iterations: u64 },
}

impl LeakageBound {
    /// Constant zero bound – function leaks nothing.
    pub fn zero() -> Self {
        Self {
            kind: LeakageBoundKind::Constant(0.0),
            worst_case_bits: 0.0,
            per_set_leakage: BTreeMap::new(),
            is_tight: true,
            derivation: "zero leakage".into(),
        }
    }

    /// Constant bound.
    pub fn constant(bits: f64) -> Self {
        assert!(bits >= 0.0, "leakage bound must be non-negative");
        Self {
            kind: LeakageBoundKind::Constant(bits),
            worst_case_bits: bits,
            per_set_leakage: BTreeMap::new(),
            is_tight: false,
            derivation: format!("constant({:.4})", bits),
        }
    }

    /// Per-set bound.
    pub fn per_set(per_set: BTreeMap<u32, f64>) -> Self {
        let worst = per_set.values().sum::<f64>();
        Self {
            kind: LeakageBoundKind::PerSet(per_set.clone()),
            worst_case_bits: worst,
            per_set_leakage: per_set,
            is_tight: false,
            derivation: "per-set bound".into(),
        }
    }

    /// Evaluate this bound on a given abstract cache state.
    pub fn evaluate(&self, state: &AbstractCacheState) -> f64 {
        match &self.kind {
            LeakageBoundKind::Constant(c) => *c,
            LeakageBoundKind::PerSet(map) => {
                map.iter()
                    .map(|(&idx, &bound)| {
                        let _set_state = state.get_set(idx);
                        bound
                    })
                    .sum()
            }
            LeakageBoundKind::Symbolic { .. } => self.worst_case_bits,
            LeakageBoundKind::Composed { components, operation } => match operation {
                CompositionOp::Sum | CompositionOp::SumWithTransform => {
                    components.iter().sum()
                }
                CompositionOp::Max => {
                    components.iter().cloned().fold(0.0_f64, f64::max)
                }
                CompositionOp::LoopUnroll { iterations } => {
                    let per_iter: f64 = components.iter().sum();
                    per_iter * (*iterations as f64)
                }
            },
        }
    }

    /// Add two leakage bounds (sequential composition).
    pub fn add(&self, other: &LeakageBound) -> LeakageBound {
        let combined_worst = self.worst_case_bits + other.worst_case_bits;
        let mut per_set = self.per_set_leakage.clone();
        for (&idx, &val) in &other.per_set_leakage {
            *per_set.entry(idx).or_insert(0.0) += val;
        }
        LeakageBound {
            kind: LeakageBoundKind::Composed {
                components: vec![self.worst_case_bits, other.worst_case_bits],
                operation: CompositionOp::Sum,
            },
            worst_case_bits: combined_worst,
            per_set_leakage: per_set,
            is_tight: self.is_tight && other.is_tight,
            derivation: format!("({}) + ({})", self.derivation, other.derivation),
        }
    }

    /// Max of two leakage bounds (conditional composition).
    pub fn max(&self, other: &LeakageBound) -> LeakageBound {
        let combined_worst = self.worst_case_bits.max(other.worst_case_bits);
        let mut per_set = BTreeMap::new();
        let all_sets: Vec<u32> = self
            .per_set_leakage
            .keys()
            .chain(other.per_set_leakage.keys())
            .copied()
            .collect();
        for idx in all_sets {
            let a = self.per_set_leakage.get(&idx).copied().unwrap_or(0.0);
            let b = other.per_set_leakage.get(&idx).copied().unwrap_or(0.0);
            per_set.insert(idx, a.max(b));
        }
        LeakageBound {
            kind: LeakageBoundKind::Composed {
                components: vec![self.worst_case_bits, other.worst_case_bits],
                operation: CompositionOp::Max,
            },
            worst_case_bits: combined_worst,
            per_set_leakage: per_set,
            is_tight: false,
            derivation: format!("max({}, {})", self.derivation, other.derivation),
        }
    }

    /// Scale by a constant (loop unrolling).
    pub fn scale(&self, factor: u64) -> LeakageBound {
        let scaled_worst = self.worst_case_bits * factor as f64;
        let per_set: BTreeMap<u32, f64> = self
            .per_set_leakage
            .iter()
            .map(|(&k, &v)| (k, v * factor as f64))
            .collect();
        LeakageBound {
            kind: LeakageBoundKind::Composed {
                components: vec![self.worst_case_bits],
                operation: CompositionOp::LoopUnroll { iterations: factor },
            },
            worst_case_bits: scaled_worst,
            per_set_leakage: per_set,
            is_tight: false,
            derivation: format!("{} × ({})", factor, self.derivation),
        }
    }

    /// Whether this bound is dominated by `other` (pointwise ≤).
    pub fn dominated_by(&self, other: &LeakageBound) -> bool {
        self.worst_case_bits <= other.worst_case_bits
    }
}

impl PartialEq for LeakageBound {
    fn eq(&self, other: &Self) -> bool {
        (self.worst_case_bits - other.worst_case_bits).abs() < 1e-12
            && self.kind == other.kind
    }
}

impl fmt::Display for LeakageBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} bits", self.worst_case_bits)?;
        if self.is_tight {
            write!(f, " (tight)")?;
        }
        if !self.per_set_leakage.is_empty() {
            write!(f, " [across {} sets]", self.per_set_leakage.len())?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Contract precondition / postcondition
// ---------------------------------------------------------------------------

/// Precondition that must hold for a contract to apply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractPrecondition {
    /// Symbolic name for the precondition.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Required cache configuration parameters.
    pub required_cache_config: Option<CacheGeometryInfo>,
    /// Required security labels on inputs.
    pub security_labels: IndexMap<String, SecurityLevel>,
    /// Minimum associativity required.
    pub min_associativity: Option<u32>,
    /// Maximum speculation window assumed.
    pub max_speculation_window: Option<u32>,
    /// Whether alignment assumptions are needed.
    pub alignment_required: bool,
    /// Additional symbolic constraints.
    pub constraints: Vec<String>,
}

impl ContractPrecondition {
    /// Trivial precondition – always satisfied.
    pub fn trivial() -> Self {
        Self {
            name: "trivial".into(),
            description: "No preconditions".into(),
            required_cache_config: None,
            security_labels: IndexMap::new(),
            min_associativity: None,
            max_speculation_window: None,
            alignment_required: false,
            constraints: Vec::new(),
        }
    }

    /// Check whether a cache configuration satisfies the precondition.
    pub fn satisfied_by_config(&self, geom: &CacheGeometryInfo) -> bool {
        if let Some(ref req) = self.required_cache_config {
            if geom.num_sets != req.num_sets || geom.associativity != req.associativity {
                return false;
            }
        }
        if let Some(min_assoc) = self.min_associativity {
            if geom.associativity < min_assoc {
                return false;
            }
        }
        true
    }

    /// Merge two preconditions (intersection – must satisfy both).
    pub fn merge(&self, other: &ContractPrecondition) -> ContractPrecondition {
        let mut labels = self.security_labels.clone();
        for (k, v) in &other.security_labels {
            labels.insert(k.clone(), v.clone());
        }
        let mut constraints = self.constraints.clone();
        constraints.extend(other.constraints.iter().cloned());

        ContractPrecondition {
            name: format!("{} ∧ {}", self.name, other.name),
            description: format!("{} AND {}", self.description, other.description),
            required_cache_config: self
                .required_cache_config
                .clone()
                .or_else(|| other.required_cache_config.clone()),
            security_labels: labels,
            min_associativity: match (self.min_associativity, other.min_associativity) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            },
            max_speculation_window: match (self.max_speculation_window, other.max_speculation_window) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            alignment_required: self.alignment_required || other.alignment_required,
            constraints,
        }
    }
}

impl fmt::Display for ContractPrecondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pre({})", self.name)?;
        if let Some(min) = self.min_associativity {
            write!(f, " [assoc ≥ {}]", min)?;
        }
        if let Some(win) = self.max_speculation_window {
            write!(f, " [spec_win ≤ {}]", win)?;
        }
        Ok(())
    }
}

/// Postcondition guaranteed by a contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractPostcondition {
    /// Symbolic name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Guaranteed output security labels.
    pub output_labels: IndexMap<String, SecurityLevel>,
    /// Whether the function is guaranteed constant-time (0 leakage).
    pub constant_time: bool,
    /// Sets that are guaranteed untouched.
    pub untouched_sets: Vec<u32>,
    /// Additional guarantees.
    pub guarantees: Vec<String>,
}

impl ContractPostcondition {
    /// Postcondition for a constant-time function.
    pub fn constant_time() -> Self {
        Self {
            name: "constant-time".into(),
            description: "Function executes in constant time".into(),
            output_labels: IndexMap::new(),
            constant_time: true,
            untouched_sets: Vec::new(),
            guarantees: vec!["constant-time execution".into()],
        }
    }

    /// Trivial postcondition – no guarantees beyond the bound.
    pub fn trivial() -> Self {
        Self {
            name: "trivial".into(),
            description: "No additional guarantees".into(),
            output_labels: IndexMap::new(),
            constant_time: false,
            untouched_sets: Vec::new(),
            guarantees: Vec::new(),
        }
    }
}

impl fmt::Display for ContractPostcondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "post({})", self.name)?;
        if self.constant_time {
            write!(f, " [CT]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Contract strength
// ---------------------------------------------------------------------------

/// How strong (precise) the contract is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ContractStrength {
    /// The bound is exact – it can be attained.
    Exact,
    /// The bound is a provable upper bound but may not be tight.
    UpperBound,
    /// The bound is an approximation (heuristic, may not be sound).
    Approximate,
}

impl ContractStrength {
    /// Weaken: composing two contracts yields the weaker strength.
    pub fn compose(self, other: Self) -> Self {
        self.max(other)
    }

    pub fn is_sound(self) -> bool {
        matches!(self, ContractStrength::Exact | ContractStrength::UpperBound)
    }

    pub fn label(self) -> &'static str {
        match self {
            ContractStrength::Exact => "exact",
            ContractStrength::UpperBound => "upper-bound",
            ContractStrength::Approximate => "approximate",
        }
    }
}

impl fmt::Display for ContractStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Contract metadata
// ---------------------------------------------------------------------------

/// Metadata attached to a contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractMetadata {
    /// When the contract was generated.
    pub timestamp: String,
    /// Tool version used.
    pub tool_version: String,
    /// Git commit of the analysed binary.
    pub binary_git_hash: Option<String>,
    /// Path to the analysed binary.
    pub binary_path: Option<String>,
    /// Analysis time in milliseconds.
    pub analysis_time_ms: u64,
    /// Number of abstract interpretation iterations.
    pub iterations: u64,
    /// Widening applied?
    pub widening_applied: bool,
    /// Additional key-value annotations.
    pub annotations: IndexMap<String, String>,
}

impl ContractMetadata {
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            binary_git_hash: None,
            binary_path: None,
            analysis_time_ms: 0,
            iterations: 0,
            widening_applied: false,
            annotations: IndexMap::new(),
        }
    }

    pub fn with_binary(mut self, path: &str, git_hash: Option<&str>) -> Self {
        self.binary_path = Some(path.to_string());
        self.binary_git_hash = git_hash.map(String::from);
        self
    }

    pub fn with_analysis_stats(mut self, time_ms: u64, iterations: u64, widening: bool) -> Self {
        self.analysis_time_ms = time_ms;
        self.iterations = iterations;
        self.widening_applied = widening;
        self
    }
}

impl Default for ContractMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ContractMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "generated={}", self.timestamp)?;
        if let Some(ref hash) = self.binary_git_hash {
            write!(f, " git={}", &hash[..8.min(hash.len())])?;
        }
        write!(f, " analysis={}ms iters={}", self.analysis_time_ms, self.iterations)
    }
}

// ---------------------------------------------------------------------------
// LeakageContract – the main type
// ---------------------------------------------------------------------------

/// A certified leakage contract for a single function.
///
/// ```text
/// Contract_f = (τ_f, B_f, pre, post, strength, metadata)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageContract {
    /// Unique identifier of the function.
    pub function_id: FunctionId,
    /// Human-readable function name.
    pub function_name: String,
    /// The cache transformer: how *f* changes abstract cache state.
    pub cache_transformer: CacheTransformer,
    /// The leakage bound: worst-case bits leaked as a function of initial state.
    pub leakage_bound: LeakageBound,
    /// Preconditions under which the contract is valid.
    pub precondition: ContractPrecondition,
    /// Postconditions guaranteed by the contract.
    pub postcondition: ContractPostcondition,
    /// Strength / precision of the bound.
    pub strength: ContractStrength,
    /// Metadata (timestamps, tool version, etc.).
    pub metadata: ContractMetadata,
    /// Content hash for integrity checking.
    pub content_hash: Option<String>,
}

impl LeakageContract {
    /// Create a new contract.
    pub fn new(
        function_id: FunctionId,
        function_name: impl Into<String>,
        cache_transformer: CacheTransformer,
        leakage_bound: LeakageBound,
    ) -> Self {
        Self {
            function_id,
            function_name: function_name.into(),
            cache_transformer,
            leakage_bound,
            precondition: ContractPrecondition::trivial(),
            postcondition: ContractPostcondition::trivial(),
            strength: ContractStrength::UpperBound,
            metadata: ContractMetadata::new(),
            content_hash: None,
        }
    }

    /// Builder: set precondition.
    pub fn with_precondition(mut self, pre: ContractPrecondition) -> Self {
        self.precondition = pre;
        self
    }

    /// Builder: set postcondition.
    pub fn with_postcondition(mut self, post: ContractPostcondition) -> Self {
        self.postcondition = post;
        self
    }

    /// Builder: set strength.
    pub fn with_strength(mut self, s: ContractStrength) -> Self {
        self.strength = s;
        self
    }

    /// Builder: set metadata.
    pub fn with_metadata(mut self, m: ContractMetadata) -> Self {
        self.metadata = m;
        self
    }

    /// Contract for a function that leaks zero bits (constant-time).
    pub fn constant_time(function_id: FunctionId, name: impl Into<String>) -> Self {
        Self {
            function_id,
            function_name: name.into(),
            cache_transformer: CacheTransformer::identity(),
            leakage_bound: LeakageBound::zero(),
            precondition: ContractPrecondition::trivial(),
            postcondition: ContractPostcondition::constant_time(),
            strength: ContractStrength::Exact,
            metadata: ContractMetadata::new(),
            content_hash: None,
        }
    }

    /// Worst-case leakage in bits.
    pub fn worst_case_bits(&self) -> f64 {
        self.leakage_bound.worst_case_bits
    }

    /// Whether the function is constant-time (zero leakage).
    pub fn is_constant_time(&self) -> bool {
        self.leakage_bound.worst_case_bits == 0.0
    }

    /// Evaluate the leakage bound on a given abstract state.
    pub fn evaluate(&self, state: &AbstractCacheState) -> f64 {
        self.leakage_bound.evaluate(state)
    }

    /// Apply the cache transformer to the given abstract state.
    pub fn transform(&self, state: &AbstractCacheState) -> AbstractCacheState {
        self.cache_transformer.apply(state)
    }

    /// Compute a SHA-256 content hash for integrity verification.
    pub fn compute_hash(&self) -> String {
        use sha2::{Sha256, Digest};
        let json = serde_json::to_string(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Store the content hash.
    pub fn seal(&mut self) {
        self.content_hash = Some(self.compute_hash());
    }

    /// Verify content hash integrity.
    pub fn verify_integrity(&self) -> bool {
        match &self.content_hash {
            Some(stored) => {
                let mut copy = self.clone();
                copy.content_hash = None;
                let expected = copy.compute_hash();
                *stored == expected
            }
            None => true, // no hash stored ⇒ trivially valid
        }
    }

    /// Number of cache sets touched by the transformer.
    pub fn touched_sets(&self) -> usize {
        self.cache_transformer.modified_sets()
    }

    /// Whether this contract is sound (Exact or UpperBound).
    pub fn is_sound(&self) -> bool {
        self.strength.is_sound()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Compare two contracts: returns the delta in worst-case bits.
    pub fn delta(&self, other: &LeakageContract) -> f64 {
        other.worst_case_bits() - self.worst_case_bits()
    }

    /// Whether this contract is strictly tighter than `other`.
    pub fn tighter_than(&self, other: &LeakageContract) -> bool {
        self.worst_case_bits() < other.worst_case_bits()
    }

    /// Brief one-line summary.
    pub fn summary(&self) -> String {
        format!(
            "{} : leaks ≤ {:.4} bits [{}]",
            self.function_name,
            self.worst_case_bits(),
            self.strength.label(),
        )
    }
}

impl PartialEq for LeakageContract {
    fn eq(&self, other: &Self) -> bool {
        self.function_id == other.function_id
            && self.cache_transformer == other.cache_transformer
            && self.leakage_bound == other.leakage_bound
            && self.strength == other.strength
    }
}

impl fmt::Display for LeakageContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Contract({}):", self.function_name)?;
        writeln!(f, "  τ: {}", self.cache_transformer)?;
        writeln!(f, "  B: {}", self.leakage_bound)?;
        writeln!(f, "  strength: {}", self.strength)?;
        writeln!(f, "  {}", self.precondition)?;
        writeln!(f, "  {}", self.postcondition)?;
        writeln!(f, "  {}", self.metadata)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_geometry() -> CacheGeometryInfo {
        CacheGeometryInfo {
            num_sets: 64,
            associativity: 8,
            line_size_bits: 6,
        }
    }

    #[test]
    fn test_cache_line_state_join() {
        let a = CacheLineState::Known { tag: 42 };
        let b = CacheLineState::Known { tag: 42 };
        assert_eq!(a.join(&b), CacheLineState::Known { tag: 42 });

        let c = CacheLineState::Known { tag: 99 };
        let joined = a.join(&c);
        assert!(matches!(joined, CacheLineState::OneOf(_)));

        let d = CacheLineState::Unknown;
        assert_eq!(a.join(&d), CacheLineState::Unknown);
    }

    #[test]
    fn test_cache_line_state_meet() {
        let a = CacheLineState::Known { tag: 42 };
        let b = CacheLineState::Unknown;
        assert_eq!(a.meet(&b), CacheLineState::Known { tag: 42 });

        let c = CacheLineState::Known { tag: 99 };
        assert_eq!(a.meet(&c), CacheLineState::Empty);
    }

    #[test]
    fn test_cache_set_state_lru_access() {
        let mut set = CacheSetState::new_empty(4);
        set.lru_access(10);
        set.lru_access(20);
        set.lru_access(30);
        assert_eq!(set.ways[0], CacheLineState::Known { tag: 30 });
        assert_eq!(set.ways[1], CacheLineState::Known { tag: 20 });
        assert_eq!(set.ways[2], CacheLineState::Known { tag: 10 });

        // Hit: tag 10 should move to MRU
        set.lru_access(10);
        assert_eq!(set.ways[0], CacheLineState::Known { tag: 10 });
        assert_eq!(set.ways[1], CacheLineState::Known { tag: 30 });
    }

    #[test]
    fn test_abstract_cache_state_join() {
        let geom = test_geometry();
        let mut s1 = AbstractCacheState::new_empty(geom.clone());
        let mut s2 = AbstractCacheState::new_empty(geom);

        s1.get_set_mut(5).lru_access(42);
        s2.get_set_mut(5).lru_access(99);

        let joined = s1.join(&s2);
        let set5 = joined.get_set(5);
        // MRU way should be OneOf({42, 99})
        assert!(matches!(set5.ways[0], CacheLineState::OneOf(_)));
    }

    #[test]
    fn test_cache_transformer_identity() {
        let geom = test_geometry();
        let state = AbstractCacheState::new_empty(geom);
        let tau = CacheTransformer::identity();
        assert_eq!(tau.apply(&state), state);
    }

    #[test]
    fn test_cache_transformer_apply() {
        let geom = test_geometry();
        let state = AbstractCacheState::new_empty(geom);
        let mut rules = BTreeMap::new();
        rules.insert(3, SetTransformRule::LruAccess { tag: 0xABC });
        let tau = CacheTransformer::from_rules(rules, "test");
        let result = tau.apply(&state);
        let set3 = result.get_set(3);
        assert_eq!(set3.ways[0], CacheLineState::Known { tag: 0xABC });
    }

    #[test]
    fn test_cache_transformer_compose() {
        let mut r1 = BTreeMap::new();
        r1.insert(1, SetTransformRule::LruAccess { tag: 10 });
        let t1 = CacheTransformer::from_rules(r1, "f");

        let mut r2 = BTreeMap::new();
        r2.insert(2, SetTransformRule::LruAccess { tag: 20 });
        let t2 = CacheTransformer::from_rules(r2, "g");

        let composed = t1.compose(&t2);
        assert!(!composed.is_identity);
        assert!(composed.set_transforms.contains_key(&1));
        assert!(composed.set_transforms.contains_key(&2));
    }

    #[test]
    fn test_leakage_bound_add() {
        let b1 = LeakageBound::constant(1.5);
        let b2 = LeakageBound::constant(2.3);
        let sum = b1.add(&b2);
        assert!((sum.worst_case_bits - 3.8).abs() < 1e-10);
    }

    #[test]
    fn test_leakage_bound_max() {
        let b1 = LeakageBound::constant(1.5);
        let b2 = LeakageBound::constant(2.3);
        let m = b1.max(&b2);
        assert!((m.worst_case_bits - 2.3).abs() < 1e-10);
    }

    #[test]
    fn test_leakage_bound_scale() {
        let b = LeakageBound::constant(1.7);
        let scaled = b.scale(10);
        assert!((scaled.worst_case_bits - 17.0).abs() < 1e-10);
    }

    #[test]
    fn test_contract_creation() {
        let c = LeakageContract::new(
            FunctionId(1),
            "aes_round",
            CacheTransformer::identity(),
            LeakageBound::constant(1.7),
        );
        assert!((c.worst_case_bits() - 1.7).abs() < 1e-10);
        assert!(!c.is_constant_time());
        assert_eq!(c.function_name, "aes_round");
    }

    #[test]
    fn test_constant_time_contract() {
        let c = LeakageContract::constant_time(FunctionId(2), "ct_memcmp");
        assert!(c.is_constant_time());
        assert_eq!(c.worst_case_bits(), 0.0);
        assert_eq!(c.strength, ContractStrength::Exact);
    }

    #[test]
    fn test_contract_serialization() {
        let c = LeakageContract::new(
            FunctionId(1),
            "test_func",
            CacheTransformer::identity(),
            LeakageBound::constant(2.5),
        );
        let json = c.to_json().unwrap();
        let restored = LeakageContract::from_json(&json).unwrap();
        assert_eq!(c, restored);
    }

    #[test]
    fn test_contract_hash_integrity() {
        let mut c = LeakageContract::new(
            FunctionId(1),
            "test_func",
            CacheTransformer::identity(),
            LeakageBound::constant(2.5),
        );
        c.seal();
        assert!(c.verify_integrity());
    }

    #[test]
    fn test_contract_strength_ordering() {
        assert!(ContractStrength::Exact < ContractStrength::UpperBound);
        assert!(ContractStrength::UpperBound < ContractStrength::Approximate);
    }

    #[test]
    fn test_contract_strength_compose() {
        assert_eq!(
            ContractStrength::Exact.compose(ContractStrength::UpperBound),
            ContractStrength::UpperBound
        );
        assert_eq!(
            ContractStrength::UpperBound.compose(ContractStrength::Approximate),
            ContractStrength::Approximate
        );
    }

    #[test]
    fn test_precondition_merge() {
        let p1 = ContractPrecondition {
            name: "p1".into(),
            description: "first".into(),
            required_cache_config: None,
            security_labels: IndexMap::new(),
            min_associativity: Some(4),
            max_speculation_window: Some(100),
            alignment_required: false,
            constraints: vec!["x > 0".into()],
        };
        let p2 = ContractPrecondition {
            name: "p2".into(),
            description: "second".into(),
            required_cache_config: None,
            security_labels: IndexMap::new(),
            min_associativity: Some(8),
            max_speculation_window: Some(50),
            alignment_required: true,
            constraints: vec!["y > 0".into()],
        };
        let merged = p1.merge(&p2);
        assert_eq!(merged.min_associativity, Some(8));
        assert_eq!(merged.max_speculation_window, Some(50));
        assert!(merged.alignment_required);
        assert_eq!(merged.constraints.len(), 2);
    }

    #[test]
    fn test_over_approximation() {
        let geom = test_geometry();
        let mut concrete = AbstractCacheState::new_empty(geom.clone());
        concrete.get_set_mut(0).lru_access(42);

        let abstract_state = AbstractCacheState::new_unknown(geom);
        assert!(abstract_state.over_approximates(&concrete));
    }

    #[test]
    fn test_contract_display() {
        let c = LeakageContract::new(
            FunctionId(1),
            "aes_sbox",
            CacheTransformer::identity(),
            LeakageBound::constant(1.7),
        );
        let display = format!("{}", c);
        assert!(display.contains("aes_sbox"));
        assert!(display.contains("1.7"));
    }

    #[test]
    fn test_contract_delta() {
        let c1 = LeakageContract::new(
            FunctionId(1), "f",
            CacheTransformer::identity(),
            LeakageBound::constant(1.0),
        );
        let c2 = LeakageContract::new(
            FunctionId(1), "f",
            CacheTransformer::identity(),
            LeakageBound::constant(2.0),
        );
        assert!((c1.delta(&c2) - 1.0).abs() < 1e-10);
        assert!(c1.tighter_than(&c2));
    }

    #[test]
    fn test_set_transform_access_sequence() {
        let geom = test_geometry();
        let state = AbstractCacheState::new_empty(geom);
        let mut rules = BTreeMap::new();
        rules.insert(0, SetTransformRule::AccessSequence(vec![10, 20, 30]));
        let tau = CacheTransformer::from_rules(rules, "seq");
        let result = tau.apply(&state);
        let set0 = result.get_set(0);
        assert_eq!(set0.ways[0], CacheLineState::Known { tag: 30 });
        assert_eq!(set0.ways[1], CacheLineState::Known { tag: 20 });
        assert_eq!(set0.ways[2], CacheLineState::Known { tag: 10 });
    }

    #[test]
    fn test_uncertainty_bits() {
        assert_eq!(CacheLineState::Known { tag: 1 }.uncertainty_bits(), 0.0);
        assert_eq!(CacheLineState::Empty.uncertainty_bits(), 0.0);
        assert_eq!(CacheLineState::Unknown.uncertainty_bits(), 64.0);
        let one_of = CacheLineState::OneOf(vec![1, 2, 3, 4]);
        assert!((one_of.uncertainty_bits() - 2.0).abs() < 1e-10);
    }
}

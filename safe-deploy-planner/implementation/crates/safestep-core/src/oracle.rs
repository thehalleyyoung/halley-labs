//! Compatibility oracle for the SafeStep deployment planner.
//!
//! The oracle answers queries of the form: "Are service A at version X and
//! service B at version Y compatible?" Results are cached in an LRU cache
//! and may originate from multiple prioritised sources (schema-derived,
//! user-specified, historical).  An [`OracleValidator`] checks the
//! consistency properties (transitivity, downward closure, symmetry) of the
//! information stored in the oracle.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Constraint, CoreResult, ServiceIndex, VersionIndex};

// ---------------------------------------------------------------------------
// CompatResult
// ---------------------------------------------------------------------------

/// Outcome of a compatibility query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatResult {
    /// The two service-version pairs are known to be compatible.
    Compatible,
    /// The two service-version pairs are known to be incompatible.
    Incompatible,
    /// Compatibility is unknown; `confidence` in [0.0, 1.0] indicates how
    /// close we are to a definitive answer (0 = no idea, 1 = almost certain).
    Unknown { confidence: f64 },
}

impl CompatResult {
    /// Returns `true` when the result is definitively [`Compatible`].
    pub fn is_compatible(&self) -> bool {
        matches!(self, CompatResult::Compatible)
    }

    /// Returns `true` when the result is definitively [`Incompatible`].
    pub fn is_incompatible(&self) -> bool {
        matches!(self, CompatResult::Incompatible)
    }

    /// Returns `true` when the result is [`Unknown`].
    pub fn is_unknown(&self) -> bool {
        matches!(self, CompatResult::Unknown { .. })
    }

    /// Returns the confidence level. Definitive answers return 1.0.
    pub fn confidence(&self) -> f64 {
        match self {
            CompatResult::Compatible | CompatResult::Incompatible => 1.0,
            CompatResult::Unknown { confidence } => *confidence,
        }
    }

    /// Merge two results, preferring the more confident / definitive one.
    pub fn merge(&self, other: &CompatResult) -> CompatResult {
        match (self, other) {
            (CompatResult::Compatible, _) | (_, CompatResult::Compatible) => {
                CompatResult::Compatible
            }
            (CompatResult::Incompatible, _) | (_, CompatResult::Incompatible) => {
                CompatResult::Incompatible
            }
            (
                CompatResult::Unknown { confidence: c1 },
                CompatResult::Unknown { confidence: c2 },
            ) => CompatResult::Unknown {
                confidence: c1.max(*c2),
            },
        }
    }
}

impl PartialEq for CompatResult {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CompatResult::Compatible, CompatResult::Compatible) => true,
            (CompatResult::Incompatible, CompatResult::Incompatible) => true,
            (CompatResult::Unknown { confidence: a }, CompatResult::Unknown { confidence: b }) => {
                (a - b).abs() < f64::EPSILON
            }
            _ => false,
        }
    }
}

impl Eq for CompatResult {}

impl fmt::Display for CompatResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompatResult::Compatible => write!(f, "compatible"),
            CompatResult::Incompatible => write!(f, "incompatible"),
            CompatResult::Unknown { confidence } => {
                write!(f, "unknown(confidence={confidence:.2})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// OracleQuery
// ---------------------------------------------------------------------------

/// Key for a compatibility query — always stored in canonical (sorted) order
/// so that (A,va,B,vb) and (B,vb,A,va) map to the same entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OracleQuery {
    pub service_a: ServiceIndex,
    pub version_a: VersionIndex,
    pub service_b: ServiceIndex,
    pub version_b: VersionIndex,
}

impl OracleQuery {
    /// Create a new query, automatically canonicalising the order so that
    /// `service_a <= service_b` (by index).  If the service indices are equal
    /// we further sort by version index.
    pub fn new(
        svc_a: ServiceIndex,
        ver_a: VersionIndex,
        svc_b: ServiceIndex,
        ver_b: VersionIndex,
    ) -> Self {
        if svc_a.0 < svc_b.0 || (svc_a.0 == svc_b.0 && ver_a.0 <= ver_b.0) {
            Self {
                service_a: svc_a,
                version_a: ver_a,
                service_b: svc_b,
                version_b: ver_b,
            }
        } else {
            Self {
                service_a: svc_b,
                version_a: ver_b,
                service_b: svc_a,
                version_b: ver_a,
            }
        }
    }

    /// Returns `true` when both sides refer to the same service.
    pub fn is_self_query(&self) -> bool {
        self.service_a == self.service_b
    }
}

impl fmt::Display for OracleQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({},{}) <-> ({},{})",
            self.service_a, self.version_a, self.service_b, self.version_b
        )
    }
}

// ---------------------------------------------------------------------------
// OracleCache — LRU cache
// ---------------------------------------------------------------------------

/// Least-recently-used cache for oracle query results.
///
/// Internally stored as a [`VecDeque`] of `(key, value)` pairs with the
/// most-recently-used item at the *front*.  A side [`HashMap`] provides O(1)
/// key → index lookup; indices are maintained lazily by scanning on `get`.
pub struct OracleCache {
    capacity: usize,
    entries: VecDeque<(OracleQuery, CompatResult)>,
    hits: u64,
    misses: u64,
}

impl OracleCache {
    /// Create a cache that holds at most `capacity` entries.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            entries: VecDeque::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    /// Look up `key` and, if found, promote it to the front (most-recently-used).
    pub fn get(&mut self, key: &OracleQuery) -> Option<&CompatResult> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == key) {
            self.hits += 1;
            // Promote: remove from current position and push to front.
            let entry = self.entries.remove(pos).unwrap();
            self.entries.push_front(entry);
            Some(&self.entries.front().unwrap().1)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a new entry (or update an existing one), evicting the
    /// least-recently-used item when at capacity.
    pub fn insert(&mut self, key: OracleQuery, value: CompatResult) {
        // If the key already exists, remove the old entry first.
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
            self.entries.remove(pos);
        }

        // Evict LRU (back) if at capacity.
        if self.entries.len() >= self.capacity {
            self.entries.pop_back();
        }

        self.entries.push_front((key, value));
    }

    /// Fraction of `get` calls that returned a hit.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Number of entries currently in the cache.
    pub fn cache_size(&self) -> usize {
        self.entries.len()
    }

    /// Remove every cached entry that involves the given service pair.
    pub fn invalidate(&mut self, svc_a: ServiceIndex, svc_b: ServiceIndex) {
        let (lo, hi) = if svc_a.0 <= svc_b.0 {
            (svc_a, svc_b)
        } else {
            (svc_b, svc_a)
        };
        self.entries
            .retain(|(q, _)| !(q.service_a == lo && q.service_b == hi));
    }

    /// Drop every entry.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Total number of queries (hits + misses).
    pub fn total_queries(&self) -> u64 {
        self.hits + self.misses
    }
}

impl fmt::Debug for OracleCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OracleCache")
            .field("capacity", &self.capacity)
            .field("size", &self.entries.len())
            .field("hit_rate", &self.hit_rate())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CompatibilityOracle
// ---------------------------------------------------------------------------

/// The primary interface to compatibility information.
///
/// Wraps a canonical data store with an LRU cache.  Entries can be added
/// individually or bulk-loaded from [`Constraint::Compatibility`] items.
pub struct CompatibilityOracle {
    /// Canonical ground-truth data.
    data: HashMap<OracleQuery, CompatResult>,
    /// LRU query cache.
    cache: RefCell<OracleCache>,
    /// Running total of queries answered.
    total_queries: Cell<u64>,
    /// Running total of queries that returned a definitive result.
    definitive_queries: Cell<u64>,
}

impl CompatibilityOracle {
    /// Create an empty oracle with a default cache capacity.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            cache: RefCell::new(OracleCache::new(4096)),
            total_queries: Cell::new(0),
            definitive_queries: Cell::new(0),
        }
    }

    /// Create an oracle with a specific cache capacity.
    pub fn with_cache_capacity(capacity: usize) -> Self {
        Self {
            data: HashMap::new(),
            cache: RefCell::new(OracleCache::new(capacity)),
            total_queries: Cell::new(0),
            definitive_queries: Cell::new(0),
        }
    }

    /// Query the oracle for the compatibility of two service-version pairs.
    pub fn query(
        &self,
        service_a: ServiceIndex,
        version_a: VersionIndex,
        service_b: ServiceIndex,
        version_b: VersionIndex,
    ) -> CompatResult {
        let key = OracleQuery::new(service_a, version_a, service_b, version_b);
        self.total_queries.set(self.total_queries.get() + 1);

        // 1. Try the LRU cache first.
        if let Some(result) = self.cache.borrow_mut().get(&key).cloned() {
            if !result.is_unknown() {
                self.definitive_queries
                    .set(self.definitive_queries.get() + 1);
            }
            return result;
        }

        // 2. Look up in canonical data.
        let result = self
            .data
            .get(&key)
            .cloned()
            .unwrap_or(CompatResult::Unknown { confidence: 0.0 });

        if !result.is_unknown() {
            self.definitive_queries
                .set(self.definitive_queries.get() + 1);
        }

        // 3. Populate cache.
        self.cache.borrow_mut().insert(key, result.clone());
        result
    }

    /// Record a compatibility fact.
    pub fn add_compatibility(
        &mut self,
        svc_a: ServiceIndex,
        ver_a: VersionIndex,
        svc_b: ServiceIndex,
        ver_b: VersionIndex,
        result: CompatResult,
    ) {
        let key = OracleQuery::new(svc_a, ver_a, svc_b, ver_b);
        // Invalidate cache for this service pair so stale entries are removed.
        self.cache.get_mut().invalidate(key.service_a, key.service_b);
        self.data.insert(key, result);
    }

    /// Bulk-load compatibility data from a slice of [`Constraint`]s.
    ///
    /// Only [`Constraint::Compatibility`] variants are considered; other
    /// constraint kinds are silently ignored.
    pub fn bulk_load(&mut self, constraints: &[Constraint]) {
        for constraint in constraints {
            if let Constraint::Compatibility {
                service_a,
                service_b,
                compatible_pairs,
                ..
            } = constraint
            {
                for &(ver_a, ver_b) in compatible_pairs {
                    self.add_compatibility(
                        *service_a,
                        ver_a,
                        *service_b,
                        ver_b,
                        CompatResult::Compatible,
                    );
                }
            }
        }
    }

    /// Proportion of queries that returned a definitive result.
    pub fn confidence_rate(&self) -> f64 {
        if self.total_queries.get() == 0 {
            0.0
        } else {
            self.definitive_queries.get() as f64 / self.total_queries.get() as f64
        }
    }

    /// Cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        self.cache.borrow().hit_rate()
    }

    /// Number of known compatibility facts.
    pub fn known_facts(&self) -> usize {
        self.data.len()
    }

    /// Total queries served.
    pub fn total_queries(&self) -> u64 {
        self.total_queries.get()
    }

    /// Access the underlying data map (read-only).
    pub fn data(&self) -> &HashMap<OracleQuery, CompatResult> {
        &self.data
    }

    /// Reset all counters and cached data.
    pub fn reset(&mut self) {
        self.data.clear();
        self.cache.get_mut().clear();
        self.total_queries.set(0);
        self.definitive_queries.set(0);
    }
}

impl Default for CompatibilityOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CompatibilityOracle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_queries = self.total_queries.get();
        let cache = self.cache.borrow();
        f.debug_struct("CompatibilityOracle")
            .field("known_facts", &self.data.len())
            .field("total_queries", &total_queries)
            .field("cache", &*cache)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// OracleSourceKind / OracleSource
// ---------------------------------------------------------------------------

/// Classification of an oracle source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OracleSourceKind {
    /// Derived from the service schema / API contracts.
    SchemaDerived,
    /// Manually specified by the user / operator.
    UserSpecified,
    /// Inferred from historical deployment data.
    Historical,
}

impl OracleSourceKind {
    /// Default priority for this kind (lower number = higher priority).
    fn default_priority(&self) -> u32 {
        match self {
            OracleSourceKind::UserSpecified => 0,
            OracleSourceKind::SchemaDerived => 1,
            OracleSourceKind::Historical => 2,
        }
    }

    /// Default confidence level for this kind.
    fn default_confidence(&self) -> f64 {
        match self {
            OracleSourceKind::UserSpecified => 1.0,
            OracleSourceKind::SchemaDerived => 0.95,
            OracleSourceKind::Historical => 0.75,
        }
    }

    /// Human-readable name.
    fn label(&self) -> &'static str {
        match self {
            OracleSourceKind::SchemaDerived => "schema-derived",
            OracleSourceKind::UserSpecified => "user-specified",
            OracleSourceKind::Historical => "historical",
        }
    }
}

/// A single source of compatibility data for the [`CompositeOracle`].
pub struct OracleSource {
    kind: OracleSourceKind,
    data: HashMap<OracleQuery, CompatResult>,
    priority: u32,
    confidence: f64,
}

impl OracleSource {
    /// Create a schema-derived source.
    pub fn schema_derived(data: HashMap<OracleQuery, CompatResult>) -> Self {
        let kind = OracleSourceKind::SchemaDerived;
        Self {
            kind,
            data,
            priority: kind.default_priority(),
            confidence: kind.default_confidence(),
        }
    }

    /// Create a user-specified source.
    pub fn user_specified(data: HashMap<OracleQuery, CompatResult>) -> Self {
        let kind = OracleSourceKind::UserSpecified;
        Self {
            kind,
            data,
            priority: kind.default_priority(),
            confidence: kind.default_confidence(),
        }
    }

    /// Create a historical source.
    pub fn historical(data: HashMap<OracleQuery, CompatResult>) -> Self {
        let kind = OracleSourceKind::Historical;
        Self {
            kind,
            data,
            priority: kind.default_priority(),
            confidence: kind.default_confidence(),
        }
    }

    /// Priority (lower = higher priority).
    pub fn priority(&self) -> u32 {
        self.priority
    }

    /// Look up a query in this source.
    pub fn query(&self, key: &OracleQuery) -> Option<CompatResult> {
        self.data.get(key).cloned()
    }

    /// Human-readable name for this source.
    pub fn name(&self) -> &str {
        self.kind.label()
    }

    /// The confidence level associated with answers from this source.
    pub fn confidence_level(&self) -> f64 {
        self.confidence
    }

    /// Number of facts stored.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether this source has any facts.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The kind of this source.
    pub fn kind(&self) -> OracleSourceKind {
        self.kind
    }

    /// Override the default priority.
    pub fn with_priority(mut self, p: u32) -> Self {
        self.priority = p;
        self
    }

    /// Override the default confidence.
    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }
}

impl fmt::Debug for OracleSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OracleSource")
            .field("kind", &self.kind)
            .field("facts", &self.data.len())
            .field("priority", &self.priority)
            .field("confidence", &self.confidence)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CompositeOracle
// ---------------------------------------------------------------------------

/// Combines multiple [`OracleSource`]s and answers queries by consulting
/// sources in priority order.
pub struct CompositeOracle {
    sources: Vec<OracleSource>,
    cache: RefCell<OracleCache>,
    total_queries: Cell<u64>,
}

impl CompositeOracle {
    /// Create an empty composite oracle.
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            cache: RefCell::new(OracleCache::new(4096)),
            total_queries: Cell::new(0),
        }
    }

    /// Register a source.  Sources are re-sorted by priority after each addition.
    pub fn add_source(&mut self, source: OracleSource) {
        self.sources.push(source);
        self.sources.sort_by_key(|s| s.priority());
        self.cache.get_mut().clear();
    }

    /// Query the composite oracle.  Sources are consulted in priority order;
    /// the first definitive answer (Compatible / Incompatible) wins.  If every
    /// source answers Unknown we return the highest-confidence Unknown.
    pub fn query(
        &self,
        svc_a: ServiceIndex,
        ver_a: VersionIndex,
        svc_b: ServiceIndex,
        ver_b: VersionIndex,
    ) -> CompatResult {
        let key = OracleQuery::new(svc_a, ver_a, svc_b, ver_b);
        self.total_queries.set(self.total_queries.get() + 1);

        // Try cache.
        if let Some(result) = self.cache.borrow_mut().get(&key).cloned() {
            return result;
        }

        let mut best_unknown_confidence: f64 = 0.0;

        for source in &self.sources {
            if let Some(result) = source.query(&key) {
                match &result {
                    CompatResult::Compatible | CompatResult::Incompatible => {
                        self.cache.borrow_mut().insert(key, result.clone());
                        return result;
                    }
                    CompatResult::Unknown { confidence } => {
                        // Weight the source confidence by the answer confidence.
                        let effective = confidence * source.confidence_level();
                        if effective > best_unknown_confidence {
                            best_unknown_confidence = effective;
                        }
                    }
                }
            }
        }

        let result = CompatResult::Unknown {
            confidence: best_unknown_confidence,
        };
        self.cache.borrow_mut().insert(key, result.clone());
        result
    }

    /// Number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Total queries answered.
    pub fn total_queries(&self) -> u64 {
        self.total_queries.get()
    }

    /// Cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        self.cache.borrow().hit_rate()
    }

    /// Iterate over the registered sources.
    pub fn sources(&self) -> &[OracleSource] {
        &self.sources
    }
}

impl Default for CompositeOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CompositeOracle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_queries = self.total_queries.get();
        let cache = self.cache.borrow();
        f.debug_struct("CompositeOracle")
            .field("sources", &self.sources.len())
            .field("total_queries", &total_queries)
            .field("cache", &*cache)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Consistency validation
// ---------------------------------------------------------------------------

/// Classification of a consistency violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViolationType {
    TransitivityViolation,
    DownwardClosureViolation,
    SymmetryViolation,
}

impl fmt::Display for ViolationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViolationType::TransitivityViolation => write!(f, "transitivity"),
            ViolationType::DownwardClosureViolation => write!(f, "downward-closure"),
            ViolationType::SymmetryViolation => write!(f, "symmetry"),
        }
    }
}

/// A single consistency violation found by the [`OracleValidator`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyViolation {
    pub violation_type: ViolationType,
    pub description: String,
    pub service_a: ServiceIndex,
    pub version_a: VersionIndex,
    pub service_b: ServiceIndex,
    pub version_b: VersionIndex,
}

impl fmt::Display for ConsistencyViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] ({},{}) vs ({},{}): {}",
            self.violation_type,
            self.service_a,
            self.version_a,
            self.service_b,
            self.version_b,
            self.description
        )
    }
}

/// Validates consistency properties of the data in a [`CompatibilityOracle`].
pub struct OracleValidator {
    /// Maximum number of violations to collect per check (0 = unlimited).
    max_violations: usize,
}

impl OracleValidator {
    /// Create a validator with no violation cap.
    pub fn new() -> Self {
        Self { max_violations: 0 }
    }

    /// Create a validator that stops after `n` violations per check.
    pub fn with_limit(n: usize) -> Self {
        Self { max_violations: n }
    }

    // ----- transitivity ---------------------------------------------------

    /// Check *transitivity*: for every triple of services (A, B, C), if
    /// (A, v_a) is compatible with (B, v_b) and (B, v_b) is compatible with
    /// (C, v_c), then (A, v_a) should also be compatible with (C, v_c).
    ///
    /// Violations are reported when (A, v_a)↔(B, v_b) and (B, v_b)↔(C, v_c)
    /// are both [`Compatible`] but (A, v_a)↔(C, v_c) is [`Incompatible`].
    pub fn validate_transitivity(
        &self,
        oracle: &mut CompatibilityOracle,
        services: &[(ServiceIndex, Vec<VersionIndex>)],
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();

        for (i, (svc_a, vers_a)) in services.iter().enumerate() {
            for (j, (svc_b, vers_b)) in services.iter().enumerate() {
                if j <= i {
                    continue;
                }
                for (k, (svc_c, vers_c)) in services.iter().enumerate() {
                    if k <= j {
                        continue;
                    }
                    for &va in vers_a {
                        for &vb in vers_b {
                            let ab =
                                oracle.query(*svc_a, va, *svc_b, vb);
                            if !ab.is_compatible() {
                                continue;
                            }
                            for &vc in vers_c {
                                let bc = oracle.query(*svc_b, vb, *svc_c, vc);
                                if !bc.is_compatible() {
                                    continue;
                                }
                                let ac = oracle.query(*svc_a, va, *svc_c, vc);
                                if ac.is_incompatible() {
                                    violations.push(ConsistencyViolation {
                                        violation_type: ViolationType::TransitivityViolation,
                                        description: format!(
                                            "({},{}) compat ({},{}) and ({},{}) compat ({},{}) \
                                             but ({},{}) incompat ({},{})",
                                            svc_a, va, svc_b, vb, svc_b, vb, svc_c, vc, svc_a,
                                            va, svc_c, vc
                                        ),
                                        service_a: *svc_a,
                                        version_a: va,
                                        service_b: *svc_c,
                                        version_b: vc,
                                    });
                                    if self.max_violations > 0
                                        && violations.len() >= self.max_violations
                                    {
                                        return violations;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        violations
    }

    // ----- downward closure -----------------------------------------------

    /// Check *downward closure*: if (A, v_a2) is compatible with (B, v_b2),
    /// then for every v_a1 ≤ v_a2 and v_b1 ≤ v_b2, (A, v_a1) should also be
    /// compatible with (B, v_b1).
    ///
    /// This captures the intuition that *older* versions tend to be
    /// compatible with each other if newer ones are.
    pub fn validate_downward_closure(
        &self,
        oracle: &mut CompatibilityOracle,
        services: &[(ServiceIndex, Vec<VersionIndex>)],
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();

        for (i, (svc_a, vers_a)) in services.iter().enumerate() {
            for (j, (svc_b, vers_b)) in services.iter().enumerate() {
                if j <= i {
                    continue;
                }

                // For every compatible pair at higher versions, check all
                // lower pairs.
                for (idx_a2, &va2) in vers_a.iter().enumerate() {
                    for (idx_b2, &vb2) in vers_b.iter().enumerate() {
                        let upper =
                            oracle.query(*svc_a, va2, *svc_b, vb2);
                        if !upper.is_compatible() {
                            continue;
                        }
                        // Check all pairs with versions ≤ the compatible pair.
                        for &va1 in &vers_a[..=idx_a2] {
                            for &vb1 in &vers_b[..=idx_b2] {
                                if va1 == va2 && vb1 == vb2 {
                                    continue;
                                }
                                let lower = oracle.query(*svc_a, va1, *svc_b, vb1);
                                if lower.is_incompatible() {
                                    violations.push(ConsistencyViolation {
                                        violation_type:
                                            ViolationType::DownwardClosureViolation,
                                        description: format!(
                                            "({},{}) compat ({},{}) but lower pair \
                                             ({},{}) incompat ({},{})",
                                            svc_a, va2, svc_b, vb2, svc_a, va1, svc_b, vb1
                                        ),
                                        service_a: *svc_a,
                                        version_a: va1,
                                        service_b: *svc_b,
                                        version_b: vb1,
                                    });
                                    if self.max_violations > 0
                                        && violations.len() >= self.max_violations
                                    {
                                        return violations;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        violations
    }

    // ----- symmetry -------------------------------------------------------

    /// Check *symmetry*: compatibility must be bidirectional.  That is,
    /// query(A, va, B, vb) should equal query(B, vb, A, va).
    ///
    /// Because [`OracleQuery::new`] canonicalises order this should hold
    /// automatically when using the oracle API.  This check validates the
    /// underlying data map directly in case entries were inserted without
    /// canonicalisation.
    pub fn validate_symmetry(
        &self,
        oracle: &mut CompatibilityOracle,
        services: &[(ServiceIndex, Vec<VersionIndex>)],
    ) -> Vec<ConsistencyViolation> {
        let mut violations = Vec::new();

        for (i, (svc_a, vers_a)) in services.iter().enumerate() {
            for (j, (svc_b, vers_b)) in services.iter().enumerate() {
                if j <= i {
                    continue;
                }
                for &va in vers_a {
                    for &vb in vers_b {
                        let forward =
                            oracle.query(*svc_a, va, *svc_b, vb);
                        let backward =
                            oracle.query(*svc_b, vb, *svc_a, va);

                        let mismatch = match (&forward, &backward) {
                            (CompatResult::Compatible, CompatResult::Compatible) => false,
                            (CompatResult::Incompatible, CompatResult::Incompatible) => false,
                            (
                                CompatResult::Unknown { confidence: c1 },
                                CompatResult::Unknown { confidence: c2 },
                            ) => (c1 - c2).abs() > f64::EPSILON,
                            _ => true,
                        };

                        if mismatch {
                            violations.push(ConsistencyViolation {
                                violation_type: ViolationType::SymmetryViolation,
                                description: format!(
                                    "query({},{},{},{})={} but \
                                     query({},{},{},{})={}",
                                    svc_a, va, svc_b, vb, forward, svc_b, vb, svc_a, va, backward
                                ),
                                service_a: *svc_a,
                                version_a: va,
                                service_b: *svc_b,
                                version_b: vb,
                            });
                            if self.max_violations > 0
                                && violations.len() >= self.max_violations
                            {
                                return violations;
                            }
                        }
                    }
                }
            }
        }

        violations
    }

    /// Run all three consistency checks and return the aggregated violations.
    pub fn validate_all(
        &self,
        oracle: &mut CompatibilityOracle,
        services: &[(ServiceIndex, Vec<VersionIndex>)],
    ) -> Vec<ConsistencyViolation> {
        let mut all = self.validate_symmetry(oracle, services);
        all.extend(self.validate_transitivity(oracle, services));
        all.extend(self.validate_downward_closure(oracle, services));
        all
    }
}

impl Default for OracleValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for OracleValidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OracleValidator")
            .field("max_violations", &self.max_violations)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ---------------------------------------------------------------

    fn si(n: u16) -> ServiceIndex {
        ServiceIndex(n)
    }
    fn vi(n: u16) -> VersionIndex {
        VersionIndex(n)
    }

    fn make_oracle_with_triangle() -> CompatibilityOracle {
        let mut oracle = CompatibilityOracle::new();
        // A(0)↔B(0): compatible
        oracle.add_compatibility(si(0), vi(0), si(1), vi(0), CompatResult::Compatible);
        // B(0)↔C(0): compatible
        oracle.add_compatibility(si(1), vi(0), si(2), vi(0), CompatResult::Compatible);
        // A(0)↔C(0): compatible (transitive-consistent)
        oracle.add_compatibility(si(0), vi(0), si(2), vi(0), CompatResult::Compatible);
        oracle
    }

    // Tests ----------------------------------------------------------------

    #[test]
    fn test_compat_result_properties() {
        assert!(CompatResult::Compatible.is_compatible());
        assert!(!CompatResult::Compatible.is_incompatible());
        assert!(!CompatResult::Compatible.is_unknown());
        assert_eq!(CompatResult::Compatible.confidence(), 1.0);

        assert!(CompatResult::Incompatible.is_incompatible());
        assert_eq!(CompatResult::Incompatible.confidence(), 1.0);

        let u = CompatResult::Unknown { confidence: 0.42 };
        assert!(u.is_unknown());
        assert!((u.confidence() - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compat_result_merge() {
        let c = CompatResult::Compatible;
        let i = CompatResult::Incompatible;
        let u1 = CompatResult::Unknown { confidence: 0.3 };
        let u2 = CompatResult::Unknown { confidence: 0.8 };

        // Compatible wins over Unknown.
        assert!(c.merge(&u1).is_compatible());
        // Incompatible wins over Unknown.
        assert!(i.merge(&u2).is_incompatible());
        // Two unknowns → keep higher confidence.
        let merged = u1.merge(&u2);
        assert!(merged.is_unknown());
        assert!((merged.confidence() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_oracle_query_canonicalisation() {
        let q1 = OracleQuery::new(si(3), vi(1), si(1), vi(2));
        let q2 = OracleQuery::new(si(1), vi(2), si(3), vi(1));
        assert_eq!(q1, q2);
        // The lower service index should always be first.
        assert_eq!(q1.service_a, si(1));
        assert_eq!(q1.service_b, si(3));
    }

    #[test]
    fn test_oracle_cache_lru_eviction() {
        let mut cache = OracleCache::new(3);
        let q1 = OracleQuery::new(si(0), vi(0), si(1), vi(0));
        let q2 = OracleQuery::new(si(0), vi(0), si(1), vi(1));
        let q3 = OracleQuery::new(si(0), vi(0), si(1), vi(2));
        let q4 = OracleQuery::new(si(0), vi(0), si(1), vi(3));

        cache.insert(q1.clone(), CompatResult::Compatible);
        cache.insert(q2.clone(), CompatResult::Incompatible);
        cache.insert(q3.clone(), CompatResult::Compatible);
        assert_eq!(cache.cache_size(), 3);

        // Access q1 to promote it.
        assert!(cache.get(&q1).unwrap().is_compatible());

        // Insert q4 — should evict q2 (LRU, at the back after q1 was promoted).
        cache.insert(q4.clone(), CompatResult::Compatible);
        assert_eq!(cache.cache_size(), 3);

        // q2 should be gone.
        assert!(cache.get(&q2).is_none());
        // q1, q3, q4 should still be present.
        assert!(cache.get(&q1).is_some());
        assert!(cache.get(&q3).is_some());
        assert!(cache.get(&q4).is_some());
    }

    #[test]
    fn test_oracle_cache_hit_rate() {
        let mut cache = OracleCache::new(16);
        let q1 = OracleQuery::new(si(0), vi(0), si(1), vi(0));
        cache.insert(q1.clone(), CompatResult::Compatible);

        // One hit, one miss.
        let _ = cache.get(&q1);
        let q2 = OracleQuery::new(si(0), vi(1), si(1), vi(1));
        let _ = cache.get(&q2);

        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
        assert_eq!(cache.total_queries(), 2);
    }

    #[test]
    fn test_compatibility_oracle_basic_flow() {
        let mut oracle = CompatibilityOracle::new();
        oracle.add_compatibility(si(0), vi(0), si(1), vi(0), CompatResult::Compatible);
        oracle.add_compatibility(si(0), vi(1), si(1), vi(0), CompatResult::Incompatible);

        assert!(oracle.query(si(0), vi(0), si(1), vi(0)).is_compatible());
        assert!(oracle.query(si(1), vi(0), si(0), vi(0)).is_compatible()); // symmetric
        assert!(oracle.query(si(0), vi(1), si(1), vi(0)).is_incompatible());
        // Unknown pair.
        assert!(oracle.query(si(0), vi(2), si(1), vi(2)).is_unknown());
        assert_eq!(oracle.known_facts(), 2);
        assert_eq!(oracle.total_queries(), 4);
    }

    #[test]
    fn test_compatibility_oracle_bulk_load() {
        use safestep_types::identifiers::{ConstraintId, Id, ConstraintTag};
        let constraints = vec![Constraint::Compatibility {
            id: ConstraintId::new(),
            service_a: si(0),
            service_b: si(1),
            compatible_pairs: vec![(vi(0), vi(0)), (vi(0), vi(1)), (vi(1), vi(1))],
        }];

        let mut oracle = CompatibilityOracle::new();
        oracle.bulk_load(&constraints);

        assert_eq!(oracle.known_facts(), 3);
        assert!(oracle.query(si(0), vi(0), si(1), vi(0)).is_compatible());
        assert!(oracle.query(si(0), vi(0), si(1), vi(1)).is_compatible());
        assert!(oracle.query(si(0), vi(1), si(1), vi(1)).is_compatible());
        // Not loaded → unknown.
        assert!(oracle.query(si(0), vi(1), si(1), vi(0)).is_unknown());
    }

    #[test]
    fn test_composite_oracle_priority() {
        let mut user_data = HashMap::new();
        let key = OracleQuery::new(si(0), vi(0), si(1), vi(0));
        user_data.insert(key.clone(), CompatResult::Incompatible);

        let mut schema_data = HashMap::new();
        schema_data.insert(key.clone(), CompatResult::Compatible);

        let user_src = OracleSource::user_specified(user_data);
        let schema_src = OracleSource::schema_derived(schema_data);

        let mut composite = CompositeOracle::new();
        // Add schema first — shouldn't matter, priority sorts.
        composite.add_source(schema_src);
        composite.add_source(user_src);

        assert_eq!(composite.source_count(), 2);
        // User-specified has higher priority → Incompatible wins.
        let result = composite.query(si(0), vi(0), si(1), vi(0));
        assert!(result.is_incompatible());
    }

    #[test]
    fn test_composite_oracle_fallback_to_unknown() {
        let composite_empty = &mut CompositeOracle::new();
        let result = composite_empty.query(si(0), vi(0), si(1), vi(0));
        assert!(result.is_unknown());
        assert!((result.confidence() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_validator_symmetry_clean() {
        let mut oracle = make_oracle_with_triangle();
        let services = vec![
            (si(0), vec![vi(0)]),
            (si(1), vec![vi(0)]),
            (si(2), vec![vi(0)]),
        ];
        let validator = OracleValidator::new();
        let violations = validator.validate_symmetry(&mut oracle, &services);
        assert!(violations.is_empty(), "Expected no symmetry violations");
    }

    #[test]
    fn test_validator_transitivity_clean() {
        let mut oracle = make_oracle_with_triangle();
        let services = vec![
            (si(0), vec![vi(0)]),
            (si(1), vec![vi(0)]),
            (si(2), vec![vi(0)]),
        ];
        let validator = OracleValidator::new();
        let violations = validator.validate_transitivity(&mut oracle, &services);
        assert!(
            violations.is_empty(),
            "Expected no transitivity violations for consistent triangle"
        );
    }

    #[test]
    fn test_validator_transitivity_violation() {
        let mut oracle = CompatibilityOracle::new();
        oracle.add_compatibility(si(0), vi(0), si(1), vi(0), CompatResult::Compatible);
        oracle.add_compatibility(si(1), vi(0), si(2), vi(0), CompatResult::Compatible);
        oracle.add_compatibility(si(0), vi(0), si(2), vi(0), CompatResult::Incompatible);

        let services = vec![
            (si(0), vec![vi(0)]),
            (si(1), vec![vi(0)]),
            (si(2), vec![vi(0)]),
        ];
        let validator = OracleValidator::new();
        let violations = validator.validate_transitivity(&mut oracle, &services);
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].violation_type,
            ViolationType::TransitivityViolation
        );
    }

    #[test]
    fn test_validator_downward_closure_violation() {
        let mut oracle = CompatibilityOracle::new();
        // (A,v1) compat (B,v1) — the higher pair.
        oracle.add_compatibility(si(0), vi(1), si(1), vi(1), CompatResult::Compatible);
        // (A,v0) incompat (B,v0) — a lower pair that should be compatible under DC.
        oracle.add_compatibility(si(0), vi(0), si(1), vi(0), CompatResult::Incompatible);

        let services = vec![
            (si(0), vec![vi(0), vi(1)]),
            (si(1), vec![vi(0), vi(1)]),
        ];
        let validator = OracleValidator::new();
        let violations = validator.validate_downward_closure(&mut oracle, &services);
        assert!(!violations.is_empty());
        assert_eq!(
            violations[0].violation_type,
            ViolationType::DownwardClosureViolation
        );
    }

    #[test]
    fn test_validator_all_checks() {
        let mut oracle = make_oracle_with_triangle();
        let services = vec![
            (si(0), vec![vi(0)]),
            (si(1), vec![vi(0)]),
            (si(2), vec![vi(0)]),
        ];
        let validator = OracleValidator::new();
        let violations = validator.validate_all(&mut oracle, &services);
        assert!(
            violations.is_empty(),
            "Consistent oracle should have no violations"
        );
    }

    #[test]
    fn test_oracle_cache_invalidate() {
        let mut cache = OracleCache::new(64);
        let q1 = OracleQuery::new(si(0), vi(0), si(1), vi(0));
        let q2 = OracleQuery::new(si(0), vi(1), si(1), vi(1));
        let q3 = OracleQuery::new(si(0), vi(0), si(2), vi(0));

        cache.insert(q1.clone(), CompatResult::Compatible);
        cache.insert(q2.clone(), CompatResult::Incompatible);
        cache.insert(q3.clone(), CompatResult::Compatible);
        assert_eq!(cache.cache_size(), 3);

        // Invalidate all entries for service pair (0, 1).
        cache.invalidate(si(0), si(1));
        assert_eq!(cache.cache_size(), 1);
        assert!(cache.get(&q1).is_none());
        assert!(cache.get(&q2).is_none());
        // Entry for (0, 2) still present.
        assert!(cache.get(&q3).is_some());
    }

    #[test]
    fn test_oracle_source_properties() {
        let src = OracleSource::schema_derived(HashMap::new());
        assert_eq!(src.name(), "schema-derived");
        assert_eq!(src.kind(), OracleSourceKind::SchemaDerived);
        assert!(src.is_empty());
        assert!((src.confidence_level() - 0.95).abs() < f64::EPSILON);

        let src2 = OracleSource::historical(HashMap::new()).with_confidence(0.5);
        assert!((src2.confidence_level() - 0.5).abs() < f64::EPSILON);

        let src3 = OracleSource::user_specified(HashMap::new()).with_priority(10);
        assert_eq!(src3.priority(), 10);
    }
}

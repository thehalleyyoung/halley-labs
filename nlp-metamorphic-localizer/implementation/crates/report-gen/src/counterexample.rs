//! Counterexample database and regression-test suite export.
//!
//! Stores minimal, shrunk counterexamples discovered during metamorphic
//! testing and converts them into reusable regression tests.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Core types ──────────────────────────────────────────────────────────────

/// A single counterexample: a (potentially shrunk) input that violates a
/// metamorphic relation at a specific pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleEntry {
    pub id: Uuid,
    /// Original sentence before transformation.
    pub original_text: String,
    /// Transformed sentence that triggered the violation.
    pub transformed_text: String,
    /// Name of the metamorphic transformation applied.
    pub transformation_name: String,
    /// Name of the metamorphic relation that was violated.
    pub violated_relation: String,
    /// Pipeline stage most suspected of causing the fault.
    pub faulty_stage: Option<String>,
    /// Violation magnitude (differential at the faulty stage).
    pub violation_magnitude: f64,
    /// Whether the entry has been minimised via shrinking.
    pub is_shrunk: bool,
    /// Number of shrinking steps taken (0 if not shrunk).
    pub shrink_steps: usize,
    /// Timestamp when the counterexample was recorded.
    pub created_at: DateTime<Utc>,
    /// Arbitrary tags for categorisation.
    pub tags: Vec<String>,
    /// Per-stage differentials at time of discovery.
    pub stage_differentials: Vec<(String, f64)>,
}

impl CounterexampleEntry {
    /// Create a new counterexample entry with a fresh UUID.
    pub fn new(
        original: impl Into<String>,
        transformed: impl Into<String>,
        transformation: impl Into<String>,
        relation: impl Into<String>,
        magnitude: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            original_text: original.into(),
            transformed_text: transformed.into(),
            transformation_name: transformation.into(),
            violated_relation: relation.into(),
            faulty_stage: None,
            violation_magnitude: magnitude,
            is_shrunk: false,
            shrink_steps: 0,
            created_at: Utc::now(),
            tags: Vec::new(),
            stage_differentials: Vec::new(),
        }
    }

    /// Builder: attach a faulty stage.
    pub fn with_faulty_stage(mut self, stage: impl Into<String>) -> Self {
        self.faulty_stage = Some(stage.into());
        self
    }

    /// Builder: mark as shrunk.
    pub fn with_shrink_info(mut self, steps: usize) -> Self {
        self.is_shrunk = true;
        self.shrink_steps = steps;
        self
    }

    /// Builder: add per-stage differentials.
    pub fn with_differentials(mut self, diffs: Vec<(String, f64)>) -> Self {
        self.stage_differentials = diffs;
        self
    }
}

// ── CounterexampleDB ────────────────────────────────────────────────────────

/// In-memory counterexample database supporting insertion and query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CounterexampleDB {
    entries: Vec<CounterexampleEntry>,
    /// Index from transformation name → entry indices.
    by_transformation: HashMap<String, Vec<usize>>,
    /// Index from faulty stage → entry indices.
    by_stage: HashMap<String, Vec<usize>>,
}

impl CounterexampleDB {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a counterexample and update internal indices.
    pub fn insert(&mut self, entry: CounterexampleEntry) {
        let idx = self.entries.len();
        self.by_transformation
            .entry(entry.transformation_name.clone())
            .or_default()
            .push(idx);
        if let Some(ref stage) = entry.faulty_stage {
            self.by_stage.entry(stage.clone()).or_default().push(idx);
        }
        self.entries.push(entry);
    }

    /// Number of stored counterexamples.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Retrieve all entries.
    pub fn entries(&self) -> &[CounterexampleEntry] {
        &self.entries
    }

    /// Retrieve entries by transformation name.
    pub fn by_transformation(&self, name: &str) -> Vec<&CounterexampleEntry> {
        self.by_transformation
            .get(name)
            .map(|idxs| idxs.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// Retrieve entries by faulty stage name.
    pub fn by_stage(&self, stage: &str) -> Vec<&CounterexampleEntry> {
        self.by_stage
            .get(stage)
            .map(|idxs| idxs.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// Return a [`QueryBuilder`] for fluent querying.
    pub fn query(&self) -> QueryBuilder<'_> {
        QueryBuilder::new(self)
    }

    /// Remove duplicate counterexamples (same original + transformed text).
    pub fn deduplicate(&mut self) {
        let mut seen = std::collections::HashSet::new();
        let mut keep = Vec::new();
        for entry in self.entries.drain(..) {
            let key = (entry.original_text.clone(), entry.transformed_text.clone());
            if seen.insert(key) {
                keep.push(entry);
            }
        }
        *self = Self::new();
        for entry in keep {
            self.insert(entry);
        }
    }

    /// Export all counterexamples as a [`RegressionTestSuite`].
    pub fn to_regression_suite(&self, suite_name: impl Into<String>) -> RegressionTestSuite {
        let tests = self
            .entries
            .iter()
            .map(|e| RegressionTest {
                id: e.id,
                original_text: e.original_text.clone(),
                transformed_text: e.transformed_text.clone(),
                transformation_name: e.transformation_name.clone(),
                expected_relation: e.violated_relation.clone(),
                expected_stage: e.faulty_stage.clone(),
                expected_violation: true,
                tags: e.tags.clone(),
            })
            .collect();
        RegressionTestSuite {
            name: suite_name.into(),
            tests,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Serialise the database to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialise from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ── QueryBuilder ────────────────────────────────────────────────────────────

/// Fluent query builder for filtering counterexamples.
#[derive(Debug)]
pub struct QueryBuilder<'a> {
    db: &'a CounterexampleDB,
    transformation_filter: Option<String>,
    stage_filter: Option<String>,
    min_magnitude: Option<f64>,
    max_magnitude: Option<f64>,
    shrunk_only: bool,
    tag_filter: Option<String>,
    limit: Option<usize>,
}

impl<'a> QueryBuilder<'a> {
    fn new(db: &'a CounterexampleDB) -> Self {
        Self {
            db,
            transformation_filter: None,
            stage_filter: None,
            min_magnitude: None,
            max_magnitude: None,
            shrunk_only: false,
            tag_filter: None,
            limit: None,
        }
    }

    /// Filter by transformation name.
    pub fn transformation(mut self, name: impl Into<String>) -> Self {
        self.transformation_filter = Some(name.into());
        self
    }

    /// Filter by faulty stage.
    pub fn stage(mut self, name: impl Into<String>) -> Self {
        self.stage_filter = Some(name.into());
        self
    }

    /// Filter by minimum violation magnitude.
    pub fn min_magnitude(mut self, min: f64) -> Self {
        self.min_magnitude = Some(min);
        self
    }

    /// Filter by maximum violation magnitude.
    pub fn max_magnitude(mut self, max: f64) -> Self {
        self.max_magnitude = Some(max);
        self
    }

    /// Only return shrunk counterexamples.
    pub fn shrunk_only(mut self) -> Self {
        self.shrunk_only = true;
        self
    }

    /// Filter by tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag_filter = Some(tag.into());
        self
    }

    /// Limit the number of results.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Execute the query and collect matching entries.
    pub fn execute(&self) -> Vec<&'a CounterexampleEntry> {
        let mut results: Vec<&CounterexampleEntry> = self
            .db
            .entries
            .iter()
            .filter(|e| {
                if let Some(ref tf) = self.transformation_filter {
                    if e.transformation_name != *tf {
                        return false;
                    }
                }
                if let Some(ref sf) = self.stage_filter {
                    match &e.faulty_stage {
                        Some(s) if s == sf => {}
                        _ => return false,
                    }
                }
                if let Some(min) = self.min_magnitude {
                    if e.violation_magnitude < min {
                        return false;
                    }
                }
                if let Some(max) = self.max_magnitude {
                    if e.violation_magnitude > max {
                        return false;
                    }
                }
                if self.shrunk_only && !e.is_shrunk {
                    return false;
                }
                if let Some(ref tag) = self.tag_filter {
                    if !e.tags.contains(tag) {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Sort by violation magnitude descending (most severe first).
        results.sort_by(|a, b| {
            b.violation_magnitude
                .partial_cmp(&a.violation_magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(limit) = self.limit {
            results.truncate(limit);
        }
        results
    }
}

// ── Regression tests ────────────────────────────────────────────────────────

/// A single regression test derived from a counterexample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTest {
    pub id: Uuid,
    pub original_text: String,
    pub transformed_text: String,
    pub transformation_name: String,
    /// The MR that was originally violated – expected to still be violated
    /// if the bug hasn't been fixed.
    pub expected_relation: String,
    /// Stage expected to be faulty.
    pub expected_stage: Option<String>,
    /// Whether we expect this test to still trigger a violation.
    pub expected_violation: bool,
    pub tags: Vec<String>,
}

/// A suite of regression tests for batch execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestSuite {
    pub name: String,
    pub tests: Vec<RegressionTest>,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl RegressionTestSuite {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tests: Vec::new(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add a regression test to the suite.
    pub fn add_test(&mut self, test: RegressionTest) {
        self.tests.push(test);
    }

    /// Number of tests in the suite.
    pub fn len(&self) -> usize {
        self.tests.len()
    }

    /// Whether the suite is empty.
    pub fn is_empty(&self) -> bool {
        self.tests.is_empty()
    }

    /// Merge another suite into this one.
    pub fn merge(&mut self, other: RegressionTestSuite) {
        self.tests.extend(other.tests);
    }

    /// Serialise the suite as JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialise from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Filter tests by transformation name.
    pub fn filter_by_transformation(&self, name: &str) -> Vec<&RegressionTest> {
        self.tests
            .iter()
            .filter(|t| t.transformation_name == name)
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(orig: &str, trans: &str, t_name: &str, mag: f64) -> CounterexampleEntry {
        CounterexampleEntry::new(orig, trans, t_name, "MR-invariance", mag)
    }

    fn populated_db() -> CounterexampleDB {
        let mut db = CounterexampleDB::new();
        db.insert(
            make_entry("The cat sat", "The cat was sat", "passive", 0.8)
                .with_faulty_stage("pos"),
        );
        db.insert(
            make_entry("Dogs run fast", "Dogs ran quickly", "synonym", 0.3)
                .with_faulty_stage("ner"),
        );
        db.insert(
            make_entry("A bird flew", "A bird was flown", "passive", 1.2)
                .with_faulty_stage("pos")
                .with_shrink_info(3),
        );
        db
    }

    #[test]
    fn test_insert_and_len() {
        let db = populated_db();
        assert_eq!(db.len(), 3);
    }

    #[test]
    fn test_by_transformation() {
        let db = populated_db();
        assert_eq!(db.by_transformation("passive").len(), 2);
        assert_eq!(db.by_transformation("synonym").len(), 1);
        assert_eq!(db.by_transformation("unknown").len(), 0);
    }

    #[test]
    fn test_by_stage() {
        let db = populated_db();
        assert_eq!(db.by_stage("pos").len(), 2);
        assert_eq!(db.by_stage("ner").len(), 1);
    }

    #[test]
    fn test_query_builder_transformation() {
        let db = populated_db();
        let results = db.query().transformation("passive").execute();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_builder_magnitude() {
        let db = populated_db();
        let results = db.query().min_magnitude(0.5).execute();
        assert_eq!(results.len(), 2);
        // Should be sorted by magnitude descending.
        assert!(results[0].violation_magnitude >= results[1].violation_magnitude);
    }

    #[test]
    fn test_query_builder_shrunk_only() {
        let db = populated_db();
        let results = db.query().shrunk_only().execute();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_shrunk);
    }

    #[test]
    fn test_query_builder_limit() {
        let db = populated_db();
        let results = db.query().limit(1).execute();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_builder_combined() {
        let db = populated_db();
        let results = db
            .query()
            .transformation("passive")
            .stage("pos")
            .min_magnitude(1.0)
            .execute();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].violated_relation, "MR-invariance");
    }

    #[test]
    fn test_deduplicate() {
        let mut db = CounterexampleDB::new();
        db.insert(make_entry("A", "B", "t1", 0.5));
        db.insert(make_entry("A", "B", "t1", 0.6)); // duplicate text pair
        db.insert(make_entry("C", "D", "t2", 0.7));
        db.deduplicate();
        assert_eq!(db.len(), 2);
    }

    #[test]
    fn test_to_regression_suite() {
        let db = populated_db();
        let suite = db.to_regression_suite("reg-v1");
        assert_eq!(suite.name, "reg-v1");
        assert_eq!(suite.len(), 3);
        assert!(suite.tests.iter().all(|t| t.expected_violation));
    }

    #[test]
    fn test_regression_suite_filter() {
        let db = populated_db();
        let suite = db.to_regression_suite("reg");
        let passive_tests = suite.filter_by_transformation("passive");
        assert_eq!(passive_tests.len(), 2);
    }

    #[test]
    fn test_json_roundtrip_db() {
        let db = populated_db();
        let json = db.to_json().unwrap();
        let db2 = CounterexampleDB::from_json(&json).unwrap();
        assert_eq!(db2.len(), db.len());
    }

    #[test]
    fn test_json_roundtrip_suite() {
        let db = populated_db();
        let suite = db.to_regression_suite("reg");
        let json = suite.to_json().unwrap();
        let suite2 = RegressionTestSuite::from_json(&json).unwrap();
        assert_eq!(suite2.len(), suite.len());
    }

    #[test]
    fn test_regression_suite_merge() {
        let mut s1 = RegressionTestSuite::new("s1");
        let s2 = populated_db().to_regression_suite("s2");
        s1.merge(s2);
        assert_eq!(s1.len(), 3);
    }
}

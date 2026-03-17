//! Counterexample database for storing and querying metamorphic test violations.
//!
//! Provides persistent-style (in-memory with serialisation) storage of
//! metamorphic counterexamples, integration with the shrinking engine, and
//! query/export capabilities for regression testing.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use localization::StageLocalizationResult;
use shared_types::{LocalizerError, Result, StageId};
use shrinking::ShrinkResult;

// ── Error ───────────────────────────────────────────────────────────────────

/// Counterexample-DB specific errors.
#[derive(Debug, thiserror::Error)]
pub enum CxDbError {
    #[error("entry not found: {0}")]
    NotFound(Uuid),
    #[error("duplicate entry: {0}")]
    Duplicate(Uuid),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// ── Core entry ──────────────────────────────────────────────────────────────

/// Severity of a counterexample based on violation magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CxSeverity {
    Critical,
    High,
    Medium,
    Low,
}

impl CxSeverity {
    pub fn from_magnitude(mag: f64) -> Self {
        if mag > 2.0 {
            Self::Critical
        } else if mag > 1.0 {
            Self::High
        } else if mag > 0.5 {
            Self::Medium
        } else {
            Self::Low
        }
    }
}

/// A stored counterexample entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CxEntry {
    pub id: Uuid,
    pub original_text: String,
    pub transformed_text: String,
    pub transformation_name: String,
    pub violated_relation: String,
    pub faulty_stage: Option<String>,
    pub violation_magnitude: f64,
    pub severity: CxSeverity,
    pub is_shrunk: bool,
    pub shrink_steps: usize,
    pub stage_differentials: Vec<(String, f64)>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl CxEntry {
    /// Create a new counterexample entry.
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
            severity: CxSeverity::from_magnitude(magnitude),
            is_shrunk: false,
            shrink_steps: 0,
            stage_differentials: Vec::new(),
            tags: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Attach localization data: faulty stage and per-stage differentials.
    pub fn with_localization(mut self, stage: &str, diffs: Vec<(String, f64)>) -> Self {
        self.faulty_stage = Some(stage.to_string());
        self.stage_differentials = diffs;
        self
    }

    /// Attach shrink result information.
    pub fn with_shrink(mut self, result: &ShrinkResult) -> Self {
        self.is_shrunk = true;
        self.shrink_steps = result.shrink_steps;
        self.transformed_text = result.shrunk_text.clone();
        if let Some(ref stage) = result.faulty_stage {
            self.faulty_stage = Some(stage.clone());
        }
        self
    }
}

// ── Database ────────────────────────────────────────────────────────────────

/// In-memory counterexample database with indexed access patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CxDatabase {
    entries: IndexMap<Uuid, CxEntry>,
    /// Index: transformation name → set of entry IDs.
    idx_transformation: HashMap<String, Vec<Uuid>>,
    /// Index: faulty stage → set of entry IDs.
    idx_stage: HashMap<String, Vec<Uuid>>,
    /// Index: severity → set of entry IDs.
    idx_severity: HashMap<CxSeverity, Vec<Uuid>>,
}

impl CxDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an entry. Returns error on duplicate ID.
    pub fn insert(&mut self, entry: CxEntry) -> std::result::Result<(), CxDbError> {
        if self.entries.contains_key(&entry.id) {
            return Err(CxDbError::Duplicate(entry.id));
        }
        let id = entry.id;

        // Update indices.
        self.idx_transformation
            .entry(entry.transformation_name.clone())
            .or_default()
            .push(id);
        if let Some(ref stage) = entry.faulty_stage {
            self.idx_stage.entry(stage.clone()).or_default().push(id);
        }
        self.idx_severity
            .entry(entry.severity)
            .or_default()
            .push(id);

        self.entries.insert(id, entry);
        Ok(())
    }

    /// Insert or replace an entry.
    pub fn upsert(&mut self, entry: CxEntry) {
        let _ = self.remove(entry.id);
        let _ = self.insert(entry);
    }

    /// Remove an entry by ID.
    pub fn remove(&mut self, id: Uuid) -> std::result::Result<CxEntry, CxDbError> {
        let entry = self.entries.shift_remove(&id).ok_or(CxDbError::NotFound(id))?;
        // Remove from indices.
        remove_from_index(&mut self.idx_transformation, &entry.transformation_name, id);
        if let Some(ref stage) = entry.faulty_stage {
            remove_from_index(&mut self.idx_stage, stage, id);
        }
        remove_from_severity_index(&mut self.idx_severity, entry.severity, id);
        Ok(entry)
    }

    /// Lookup by ID.
    pub fn get(&self, id: Uuid) -> Option<&CxEntry> {
        self.entries.get(&id)
    }

    /// All entries as a slice (insertion order).
    pub fn entries(&self) -> Vec<&CxEntry> {
        self.entries.values().collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Retrieve by transformation name.
    pub fn by_transformation(&self, name: &str) -> Vec<&CxEntry> {
        self.idx_transformation
            .get(name)
            .map(|ids| ids.iter().filter_map(|id| self.entries.get(id)).collect())
            .unwrap_or_default()
    }

    /// Retrieve by faulty stage.
    pub fn by_stage(&self, stage: &str) -> Vec<&CxEntry> {
        self.idx_stage
            .get(stage)
            .map(|ids| ids.iter().filter_map(|id| self.entries.get(id)).collect())
            .unwrap_or_default()
    }

    /// Retrieve by severity.
    pub fn by_severity(&self, sev: CxSeverity) -> Vec<&CxEntry> {
        self.idx_severity
            .get(&sev)
            .map(|ids| ids.iter().filter_map(|id| self.entries.get(id)).collect())
            .unwrap_or_default()
    }

    /// Return entries whose violation magnitude exceeds a threshold, sorted
    /// descending.
    pub fn above_threshold(&self, threshold: f64) -> Vec<&CxEntry> {
        let mut results: Vec<&CxEntry> = self
            .entries
            .values()
            .filter(|e| e.violation_magnitude > threshold)
            .collect();
        results.sort_by(|a, b| {
            b.violation_magnitude
                .partial_cmp(&a.violation_magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Return only shrunk entries.
    pub fn shrunk_entries(&self) -> Vec<&CxEntry> {
        self.entries.values().filter(|e| e.is_shrunk).collect()
    }

    /// Remove duplicate entries (same original + transformed text pair).
    pub fn deduplicate(&mut self) {
        let mut seen = std::collections::HashSet::new();
        let mut to_remove = Vec::new();
        for (id, entry) in &self.entries {
            let key = (entry.original_text.clone(), entry.transformed_text.clone());
            if !seen.insert(key) {
                to_remove.push(*id);
            }
        }
        for id in to_remove {
            let _ = self.remove(id);
        }
    }

    /// Build a statistics summary.
    pub fn statistics(&self) -> CxStatistics {
        let total = self.entries.len();
        let shrunk = self.entries.values().filter(|e| e.is_shrunk).count();
        let by_transformation: HashMap<String, usize> = self
            .idx_transformation
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect();
        let by_stage: HashMap<String, usize> = self
            .idx_stage
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect();
        let mean_magnitude = if total > 0 {
            self.entries.values().map(|e| e.violation_magnitude).sum::<f64>() / total as f64
        } else {
            0.0
        };
        CxStatistics {
            total,
            shrunk,
            by_transformation,
            by_stage,
            mean_magnitude,
        }
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> std::result::Result<String, CxDbError> {
        serde_json::to_string_pretty(self).map_err(CxDbError::from)
    }

    /// Deserialise from JSON.
    pub fn from_json(json: &str) -> std::result::Result<Self, CxDbError> {
        serde_json::from_str(json).map_err(CxDbError::from)
    }
}

/// Summary statistics for the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CxStatistics {
    pub total: usize,
    pub shrunk: usize,
    pub by_transformation: HashMap<String, usize>,
    pub by_stage: HashMap<String, usize>,
    pub mean_magnitude: f64,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn remove_from_index(index: &mut HashMap<String, Vec<Uuid>>, key: &str, id: Uuid) {
    if let Some(ids) = index.get_mut(key) {
        ids.retain(|&i| i != id);
        if ids.is_empty() {
            index.remove(key);
        }
    }
}

fn remove_from_severity_index(
    index: &mut HashMap<CxSeverity, Vec<Uuid>>,
    sev: CxSeverity,
    id: Uuid,
) {
    if let Some(ids) = index.get_mut(&sev) {
        ids.retain(|&i| i != id);
        if ids.is_empty() {
            index.remove(&sev);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make(orig: &str, trans: &str, tname: &str, mag: f64) -> CxEntry {
        CxEntry::new(orig, trans, tname, "invariance", mag)
    }

    fn populated() -> CxDatabase {
        let mut db = CxDatabase::new();
        db.insert(make("cat sat", "cat was sat", "passive", 0.8).with_localization("pos", vec![]))
            .unwrap();
        db.insert(make("dogs run", "dogs ran", "synonym", 0.3).with_localization("ner", vec![]))
            .unwrap();
        db.insert(
            make("bird flew", "bird was flown", "passive", 1.5).with_localization("pos", vec![]),
        )
        .unwrap();
        db
    }

    #[test]
    fn test_insert_and_len() {
        let db = populated();
        assert_eq!(db.len(), 3);
    }

    #[test]
    fn test_duplicate_rejected() {
        let mut db = CxDatabase::new();
        let entry = make("a", "b", "t", 1.0);
        let id = entry.id;
        db.insert(entry.clone()).unwrap();
        assert!(db.insert(entry).is_err());
    }

    #[test]
    fn test_by_transformation() {
        let db = populated();
        assert_eq!(db.by_transformation("passive").len(), 2);
        assert_eq!(db.by_transformation("synonym").len(), 1);
    }

    #[test]
    fn test_by_stage() {
        let db = populated();
        assert_eq!(db.by_stage("pos").len(), 2);
        assert_eq!(db.by_stage("ner").len(), 1);
    }

    #[test]
    fn test_by_severity() {
        let db = populated();
        assert_eq!(db.by_severity(CxSeverity::Medium).len(), 1); // 0.8
        assert_eq!(db.by_severity(CxSeverity::High).len(), 1); // 1.5
        assert_eq!(db.by_severity(CxSeverity::Low).len(), 1); // 0.3
    }

    #[test]
    fn test_above_threshold() {
        let db = populated();
        let results = db.above_threshold(0.5);
        assert_eq!(results.len(), 2);
        assert!(results[0].violation_magnitude >= results[1].violation_magnitude);
    }

    #[test]
    fn test_remove() {
        let mut db = populated();
        let id = db.entries().first().unwrap().id;
        db.remove(id).unwrap();
        assert_eq!(db.len(), 2);
        assert!(db.get(id).is_none());
    }

    #[test]
    fn test_deduplicate() {
        let mut db = CxDatabase::new();
        db.insert(make("A", "B", "t1", 0.5)).unwrap();
        let mut dup = make("A", "B", "t1", 0.6);
        dup.id = Uuid::new_v4(); // different ID, same text pair
        db.insert(dup).unwrap();
        db.insert(make("C", "D", "t2", 0.7)).unwrap();
        db.deduplicate();
        assert_eq!(db.len(), 2);
    }

    #[test]
    fn test_statistics() {
        let db = populated();
        let stats = db.statistics();
        assert_eq!(stats.total, 3);
        assert_eq!(stats.by_transformation["passive"], 2);
    }

    #[test]
    fn test_json_roundtrip() {
        let db = populated();
        let json = db.to_json().unwrap();
        let db2 = CxDatabase::from_json(&json).unwrap();
        assert_eq!(db2.len(), db.len());
    }

    #[test]
    fn test_upsert() {
        let mut db = CxDatabase::new();
        let mut entry = make("a", "b", "t", 1.0);
        let id = entry.id;
        db.insert(entry.clone()).unwrap();
        entry.violation_magnitude = 2.0;
        db.upsert(entry);
        assert_eq!(db.get(id).unwrap().violation_magnitude, 2.0);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_severity_classification() {
        assert_eq!(CxSeverity::from_magnitude(3.0), CxSeverity::Critical);
        assert_eq!(CxSeverity::from_magnitude(1.5), CxSeverity::High);
        assert_eq!(CxSeverity::from_magnitude(0.7), CxSeverity::Medium);
        assert_eq!(CxSeverity::from_magnitude(0.2), CxSeverity::Low);
    }
}

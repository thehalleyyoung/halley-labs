// provenance.rs — Full provenance tracking: maps constraint variable names
// back to their originating obligation, article reference, and jurisdiction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Core types ─────────────────────────────────────────────────────────────

/// A provenance record identifying the regulatory source of a constraint
/// variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub obligation_id: String,
    pub article_reference: Option<String>,
    pub jurisdiction_id: String,
    pub description: String,
    /// Encoding phase that generated this mapping (e.g. "obligation", "ilp",
    /// "temporal").
    pub encoding_phase: String,
}

impl ProvenanceRecord {
    pub fn new(
        obligation_id: impl Into<String>,
        jurisdiction_id: impl Into<String>,
    ) -> Self {
        Self {
            obligation_id: obligation_id.into(),
            article_reference: None,
            jurisdiction_id: jurisdiction_id.into(),
            description: String::new(),
            encoding_phase: "obligation".into(),
        }
    }

    pub fn with_article(mut self, article: impl Into<String>) -> Self {
        self.article_reference = Some(article.into());
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.encoding_phase = phase.into();
        self
    }
}

// ─── ProvenanceMap ──────────────────────────────────────────────────────────

/// Maps constraint-variable names to the regulatory artefact they encode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProvenanceMap {
    /// var_name → provenance record
    entries: HashMap<String, ProvenanceRecord>,
    /// obligation_id → [var_name]  (reverse index)
    by_obligation: HashMap<String, Vec<String>>,
    /// jurisdiction_id → [var_name]  (reverse index)
    by_jurisdiction: HashMap<String, Vec<String>>,
}

impl ProvenanceMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new encoding mapping.
    pub fn record_encoding(
        &mut self,
        var_name: impl Into<String>,
        record: ProvenanceRecord,
    ) {
        let var_name = var_name.into();
        self.by_obligation
            .entry(record.obligation_id.clone())
            .or_default()
            .push(var_name.clone());
        self.by_jurisdiction
            .entry(record.jurisdiction_id.clone())
            .or_default()
            .push(var_name.clone());
        self.entries.insert(var_name, record);
    }

    /// Record a simple mapping from `var_name` to `obligation_id` and
    /// `jurisdiction_id`.
    pub fn record_simple(
        &mut self,
        var_name: impl Into<String>,
        obligation_id: impl Into<String>,
        jurisdiction_id: impl Into<String>,
    ) {
        let record =
            ProvenanceRecord::new(obligation_id, jurisdiction_id);
        self.record_encoding(var_name, record);
    }

    /// Trace a variable name back to its provenance record.
    pub fn trace_back(&self, var_name: &str) -> Option<&ProvenanceRecord> {
        self.entries.get(var_name)
    }

    /// Trace many variable names at once, returning found records.
    pub fn bulk_trace<'a>(
        &'a self,
        var_names: &[&'a str],
    ) -> Vec<(&'a str, &'a ProvenanceRecord)> {
        var_names
            .iter()
            .filter_map(|name| {
                self.entries
                    .get(*name)
                    .map(|rec| (*name, rec))
            })
            .collect()
    }

    /// All variable names associated with an obligation.
    pub fn vars_for_obligation(&self, obligation_id: &str) -> &[String] {
        self.by_obligation
            .get(obligation_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// All variable names associated with a jurisdiction.
    pub fn vars_for_jurisdiction(&self, jurisdiction_id: &str) -> &[String] {
        self.by_jurisdiction
            .get(jurisdiction_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Total number of tracked variables.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterator over all entries.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ProvenanceRecord)> {
        self.entries.iter()
    }

    /// Merge another provenance map into this one. Existing entries in `self`
    /// take precedence on collisions.
    pub fn merge(&mut self, other: &ProvenanceMap) {
        for (var_name, record) in &other.entries {
            if !self.entries.contains_key(var_name) {
                self.record_encoding(var_name.clone(), record.clone());
            }
        }
    }

    /// Return all distinct obligation IDs with at least one variable.
    pub fn obligation_ids(&self) -> Vec<&String> {
        self.by_obligation.keys().collect()
    }

    /// Return all distinct jurisdiction IDs with at least one variable.
    pub fn jurisdiction_ids(&self) -> Vec<&String> {
        self.by_jurisdiction.keys().collect()
    }

    /// Pretty-print the full provenance map.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "ProvenanceMap: {} variables, {} obligations, {} jurisdictions",
            self.entries.len(),
            self.by_obligation.len(),
            self.by_jurisdiction.len(),
        ));
        let mut sorted: Vec<_> = self.entries.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());
        for (var, rec) in sorted {
            lines.push(format!(
                "  {} → obl={}, jur={}, art={}, phase={}",
                var,
                rec.obligation_id,
                rec.jurisdiction_id,
                rec.article_reference.as_deref().unwrap_or("-"),
                rec.encoding_phase,
            ));
        }
        lines.join("\n")
    }

    /// Remove all entries for a given obligation.
    pub fn remove_obligation(&mut self, obligation_id: &str) {
        if let Some(vars) = self.by_obligation.remove(obligation_id) {
            for v in &vars {
                if let Some(rec) = self.entries.remove(v) {
                    if let Some(jvars) = self.by_jurisdiction.get_mut(&rec.jurisdiction_id) {
                        jvars.retain(|x| x != v);
                    }
                }
            }
        }
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_map() -> ProvenanceMap {
        let mut pm = ProvenanceMap::new();
        pm.record_encoding(
            "comply_art6",
            ProvenanceRecord::new("obl_art6", "EU")
                .with_article("EU-AI-Act Art. 6")
                .with_description("High-risk classification")
                .with_phase("obligation"),
        );
        pm.record_encoding(
            "comply_art52",
            ProvenanceRecord::new("obl_art52", "EU")
                .with_article("EU-AI-Act Art. 52")
                .with_description("Transparency")
                .with_phase("obligation"),
        );
        pm.record_encoding(
            "comply_ccpa",
            ProvenanceRecord::new("obl_ccpa_1", "US-CA")
                .with_description("CCPA consumer rights"),
        );
        pm
    }

    #[test]
    fn test_trace_back() {
        let pm = sample_map();
        let rec = pm.trace_back("comply_art6").unwrap();
        assert_eq!(rec.obligation_id, "obl_art6");
        assert_eq!(rec.jurisdiction_id, "EU");
        assert_eq!(
            rec.article_reference.as_deref(),
            Some("EU-AI-Act Art. 6")
        );
    }

    #[test]
    fn test_trace_back_missing() {
        let pm = sample_map();
        assert!(pm.trace_back("nonexistent").is_none());
    }

    #[test]
    fn test_bulk_trace() {
        let pm = sample_map();
        let results = pm.bulk_trace(&["comply_art6", "comply_ccpa", "missing"]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_vars_for_obligation() {
        let pm = sample_map();
        let vars = pm.vars_for_obligation("obl_art6");
        assert_eq!(vars, &["comply_art6"]);
    }

    #[test]
    fn test_vars_for_jurisdiction() {
        let pm = sample_map();
        let vars = pm.vars_for_jurisdiction("EU");
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&"comply_art6".to_string()));
        assert!(vars.contains(&"comply_art52".to_string()));
    }

    #[test]
    fn test_merge() {
        let mut pm1 = ProvenanceMap::new();
        pm1.record_simple("x", "obl1", "EU");

        let mut pm2 = ProvenanceMap::new();
        pm2.record_simple("y", "obl2", "US");
        pm2.record_simple("x", "obl_other", "UK"); // collision

        pm1.merge(&pm2);
        assert_eq!(pm1.len(), 2);
        // pm1's "x" takes precedence.
        assert_eq!(pm1.trace_back("x").unwrap().obligation_id, "obl1");
        assert_eq!(pm1.trace_back("y").unwrap().obligation_id, "obl2");
    }

    #[test]
    fn test_remove_obligation() {
        let mut pm = sample_map();
        assert_eq!(pm.len(), 3);
        pm.remove_obligation("obl_art6");
        assert_eq!(pm.len(), 2);
        assert!(pm.trace_back("comply_art6").is_none());
        assert_eq!(pm.vars_for_jurisdiction("EU").len(), 1);
    }

    #[test]
    fn test_summary() {
        let pm = sample_map();
        let s = pm.summary();
        assert!(s.contains("3 variables"));
        assert!(s.contains("comply_art6"));
    }

    #[test]
    fn test_record_simple() {
        let mut pm = ProvenanceMap::new();
        pm.record_simple("v1", "obl_x", "JP");
        let rec = pm.trace_back("v1").unwrap();
        assert_eq!(rec.obligation_id, "obl_x");
        assert_eq!(rec.jurisdiction_id, "JP");
        assert!(rec.article_reference.is_none());
    }

    #[test]
    fn test_obligation_ids() {
        let pm = sample_map();
        let ids = pm.obligation_ids();
        assert_eq!(ids.len(), 3);
    }
}

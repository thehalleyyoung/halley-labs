use crate::{ObligationId, RegulatoryState};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Represents the difference between two regulatory states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalDiff {
    pub added: BTreeSet<ObligationId>,
    pub removed: BTreeSet<ObligationId>,
    pub modified: BTreeSet<ObligationId>,
}

impl TemporalDiff {
    /// Compute the diff between two regulatory states.
    pub fn diff(before: &RegulatoryState, after: &RegulatoryState) -> Self {
        let added: BTreeSet<ObligationId> = after.obligations.difference(&before.obligations).cloned().collect();
        let removed: BTreeSet<ObligationId> = before.obligations.difference(&after.obligations).cloned().collect();
        // Modified = obligations present in both but states have different ids (version change).
        let modified = if before.id != after.id {
            before.obligations.intersection(&after.obligations).cloned().collect()
        } else {
            BTreeSet::new()
        };
        Self { added, removed, modified }
    }

    /// Apply this diff to a regulatory state to produce a new state.
    pub fn apply_diff(&self, state: &RegulatoryState) -> RegulatoryState {
        let mut obligations = state.obligations.clone();
        for id in &self.added {
            obligations.insert(id.clone());
        }
        for id in &self.removed {
            obligations.remove(id);
        }
        let new_id = format!("{}-patched", state.id);
        RegulatoryState {
            id: new_id,
            obligations,
            timestamp: state.timestamp,
        }
    }

    /// Compose two sequential diffs: first `self` then `other`.
    pub fn compose_diffs(&self, other: &TemporalDiff) -> TemporalDiff {
        let added: BTreeSet<ObligationId> = self.added.difference(&other.removed)
            .cloned()
            .chain(other.added.iter().cloned())
            .collect();
        let removed: BTreeSet<ObligationId> = self.removed.difference(&other.added)
            .cloned()
            .chain(other.removed.iter().cloned())
            .collect();
        let modified: BTreeSet<ObligationId> = self.modified.union(&other.modified)
            .filter(|id| !added.contains(*id) && !removed.contains(*id))
            .cloned()
            .collect();
        TemporalDiff { added, removed, modified }
    }

    /// Invert the diff: swap added and removed.
    pub fn invert_diff(&self) -> TemporalDiff {
        TemporalDiff {
            added: self.removed.clone(),
            removed: self.added.clone(),
            modified: self.modified.clone(),
        }
    }

    /// True if no changes.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }

    /// Impact score: total number of changes.
    pub fn impact_score(&self) -> f64 {
        (self.added.len() + self.removed.len() + self.modified.len()) as f64
    }

    /// Create an empty diff.
    pub fn new_empty() -> Self {
        Self {
            added: BTreeSet::new(),
            removed: BTreeSet::new(),
            modified: BTreeSet::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(id: &str, obls: &[&str]) -> RegulatoryState {
        let obligations = obls.iter().map(|s| s.to_string()).collect();
        RegulatoryState::with_obligations(id, obligations)
    }

    #[test]
    fn test_diff_basic() {
        let before = state("s1", &["a", "b", "c"]);
        let after = state("s2", &["b", "c", "d"]);
        let diff = TemporalDiff::diff(&before, &after);
        assert_eq!(diff.added, ["d"].iter().map(|s| s.to_string()).collect());
        assert_eq!(diff.removed, ["a"].iter().map(|s| s.to_string()).collect());
        assert!(diff.modified.contains("b"));
        assert!(diff.modified.contains("c"));
    }

    #[test]
    fn test_diff_identical() {
        let s = state("s1", &["a", "b"]);
        let diff = TemporalDiff::diff(&s, &s);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_apply_diff() {
        let before = state("s1", &["a", "b"]);
        let mut diff = TemporalDiff::new_empty();
        diff.added.insert("c".to_string());
        diff.removed.insert("a".to_string());
        let result = diff.apply_diff(&before);
        assert!(result.obligations.contains("b"));
        assert!(result.obligations.contains("c"));
        assert!(!result.obligations.contains("a"));
    }

    #[test]
    fn test_compose_diffs() {
        let mut d1 = TemporalDiff::new_empty();
        d1.added.insert("x".to_string());
        d1.removed.insert("y".to_string());

        let mut d2 = TemporalDiff::new_empty();
        d2.added.insert("z".to_string());
        d2.removed.insert("x".to_string());

        let composed = d1.compose_diffs(&d2);
        assert!(!composed.added.contains("x"));
        assert!(composed.added.contains("z"));
        assert!(composed.removed.contains("y"));
        assert!(composed.removed.contains("x"));
    }

    #[test]
    fn test_invert_diff() {
        let mut diff = TemporalDiff::new_empty();
        diff.added.insert("a".to_string());
        diff.removed.insert("b".to_string());
        let inv = diff.invert_diff();
        assert!(inv.added.contains("b"));
        assert!(inv.removed.contains("a"));
    }

    #[test]
    fn test_impact_score() {
        let mut diff = TemporalDiff::new_empty();
        diff.added.insert("a".to_string());
        diff.added.insert("b".to_string());
        diff.removed.insert("c".to_string());
        assert_eq!(diff.impact_score(), 3.0);
    }

    #[test]
    fn test_new_empty() {
        let diff = TemporalDiff::new_empty();
        assert!(diff.is_empty());
        assert_eq!(diff.impact_score(), 0.0);
    }

    #[test]
    fn test_apply_preserves_timestamp() {
        let s = state("s1", &["a"]).with_timestamp(crate::ymd(2025, 6, 1));
        let diff = TemporalDiff::new_empty();
        let result = diff.apply_diff(&s);
        assert_eq!(result.timestamp, Some(crate::ymd(2025, 6, 1)));
    }
}

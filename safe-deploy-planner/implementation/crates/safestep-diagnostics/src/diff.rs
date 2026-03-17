//! Plan diffing and comparison for deployment plan evolution.
//!
//! Provides tools to compare deployment plans, detect changes, assess safety
//! impact, format diffs for human consumption, and perform generic JSON diffing.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// DiffChangeType
// ---------------------------------------------------------------------------

/// Categorizes what kind of change occurred between two plan revisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiffChangeType {
    Added,
    Removed,
    Modified,
    Unchanged,
}

impl std::fmt::Display for DiffChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Added => write!(f, "added"),
            Self::Removed => write!(f, "removed"),
            Self::Modified => write!(f, "modified"),
            Self::Unchanged => write!(f, "unchanged"),
        }
    }
}

// ---------------------------------------------------------------------------
// StepDiff
// ---------------------------------------------------------------------------

/// A single step-level difference between two plans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepDiff {
    pub step_id: String,
    pub change_type: DiffChangeType,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

impl StepDiff {
    pub fn added(step_id: impl Into<String>, new_value: impl Into<String>) -> Self {
        Self {
            step_id: step_id.into(),
            change_type: DiffChangeType::Added,
            old_value: None,
            new_value: Some(new_value.into()),
        }
    }

    pub fn removed(step_id: impl Into<String>, old_value: impl Into<String>) -> Self {
        Self {
            step_id: step_id.into(),
            change_type: DiffChangeType::Removed,
            old_value: Some(old_value.into()),
            new_value: None,
        }
    }

    pub fn modified(
        step_id: impl Into<String>,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
    ) -> Self {
        Self {
            step_id: step_id.into(),
            change_type: DiffChangeType::Modified,
            old_value: Some(old_value.into()),
            new_value: Some(new_value.into()),
        }
    }

    pub fn unchanged(step_id: impl Into<String>, value: impl Into<String>) -> Self {
        let v = value.into();
        Self {
            step_id: step_id.into(),
            change_type: DiffChangeType::Unchanged,
            old_value: Some(v.clone()),
            new_value: Some(v),
        }
    }
}

// ---------------------------------------------------------------------------
// SafetyImpact
// ---------------------------------------------------------------------------

/// Quantified safety impact of a plan diff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafetyImpact {
    /// Net change in the number of point-of-no-return states.
    pub new_pnr_count: i64,
    /// Fractional change in envelope size (positive = larger envelope).
    pub envelope_change: f64,
    /// Aggregate risk delta – negative means safer, positive means riskier.
    pub risk_delta: f64,
}

impl SafetyImpact {
    pub fn none() -> Self {
        Self {
            new_pnr_count: 0,
            envelope_change: 0.0,
            risk_delta: 0.0,
        }
    }

    /// A change is safe when risk does not increase and no new PNR states appear.
    pub fn is_safe_change(&self) -> bool {
        self.risk_delta <= 0.0 && self.new_pnr_count <= 0
    }

    /// Human-readable severity bucket.
    pub fn severity(&self) -> &str {
        let abs = self.risk_delta.abs();
        if abs < f64::EPSILON {
            "none"
        } else if abs < 0.3 {
            "low"
        } else if abs < 0.7 {
            "medium"
        } else {
            "high"
        }
    }
}

impl Default for SafetyImpact {
    fn default() -> Self {
        Self::none()
    }
}

// ---------------------------------------------------------------------------
// PlanDiffResult
// ---------------------------------------------------------------------------

/// Full result of diffing two deployment plans.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanDiffResult {
    pub added_steps: Vec<StepDiff>,
    pub removed_steps: Vec<StepDiff>,
    pub modified_steps: Vec<StepDiff>,
    pub safety_impact: SafetyImpact,
    pub summary: String,
}

impl PlanDiffResult {
    pub fn empty() -> Self {
        Self {
            added_steps: Vec::new(),
            removed_steps: Vec::new(),
            modified_steps: Vec::new(),
            safety_impact: SafetyImpact::none(),
            summary: String::from("No changes detected"),
        }
    }

    /// Total number of non-trivial changes (added + removed + modified).
    pub fn total_changes(&self) -> usize {
        self.added_steps.len() + self.removed_steps.len() + self.modified_steps.len()
    }

    /// Whether any changes exist.
    pub fn has_changes(&self) -> bool {
        self.total_changes() > 0
    }
}

// ---------------------------------------------------------------------------
// DiffImpact
// ---------------------------------------------------------------------------

/// Semantic impact classification of a version change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiffImpact {
    Breaking,
    NonBreaking,
    Neutral,
    Unknown,
}

impl std::fmt::Display for DiffImpact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Breaking => write!(f, "breaking"),
            Self::NonBreaking => write!(f, "non-breaking"),
            Self::Neutral => write!(f, "neutral"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// VersionDiff
// ---------------------------------------------------------------------------

/// Records a version change for a single service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionDiff {
    pub service: String,
    pub old_version: String,
    pub new_version: String,
    pub impact: DiffImpact,
}

impl VersionDiff {
    pub fn new(
        service: impl Into<String>,
        old_version: impl Into<String>,
        new_version: impl Into<String>,
        impact: DiffImpact,
    ) -> Self {
        Self {
            service: service.into(),
            old_version: old_version.into(),
            new_version: new_version.into(),
            impact,
        }
    }

    /// Heuristic: if the major version changed, assume breaking.
    pub fn infer_impact(old: &str, new: &str) -> DiffImpact {
        let parse = |v: &str| -> Option<(u64, u64, u64)> {
            let v = v.strip_prefix('v').unwrap_or(v);
            let parts: Vec<&str> = v.split('.').collect();
            if parts.len() >= 3 {
                Some((
                    parts[0].parse().ok()?,
                    parts[1].parse().ok()?,
                    parts[2].parse().ok()?,
                ))
            } else {
                None
            }
        };

        match (parse(old), parse(new)) {
            (Some((om, _, _)), Some((nm, _, _))) if om != nm => DiffImpact::Breaking,
            (Some((_, om, _)), Some((_, nm, _))) if om != nm => DiffImpact::NonBreaking,
            (Some(o), Some(n)) if o == n => DiffImpact::Neutral,
            (Some(_), Some(_)) => DiffImpact::NonBreaking,
            _ => DiffImpact::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// SafetyChange
// ---------------------------------------------------------------------------

/// Describes a safety-status transition for part of a deployment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetyChange {
    pub description: String,
    pub from_status: String,
    pub to_status: String,
    pub affected_states: Vec<String>,
}

impl SafetyChange {
    pub fn new(
        description: impl Into<String>,
        from_status: impl Into<String>,
        to_status: impl Into<String>,
        affected_states: Vec<String>,
    ) -> Self {
        Self {
            description: description.into(),
            from_status: from_status.into(),
            to_status: to_status.into(),
            affected_states,
        }
    }

    /// A degradation moves from a safe status to an unsafe or pnr status.
    pub fn is_degradation(&self) -> bool {
        let safe_statuses = ["safe", "green", "ok", "healthy", "passing"];
        let unsafe_statuses = ["unsafe", "pnr", "red", "failing", "critical", "degraded", "warning"];

        let from_lower = self.from_status.to_lowercase();
        let to_lower = self.to_status.to_lowercase();

        let was_safe = safe_statuses.iter().any(|s| from_lower == *s);
        let now_unsafe = unsafe_statuses.iter().any(|s| to_lower == *s);

        was_safe && now_unsafe
    }
}

// ---------------------------------------------------------------------------
// JsonChange  (used by JsonDiff)
// ---------------------------------------------------------------------------

/// A single change detected in a generic JSON comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonChange {
    /// JSON Pointer path to the changed value (e.g. `/steps/0/id`).
    pub path: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: Option<serde_json::Value>,
    pub change_type: DiffChangeType,
}

// ---------------------------------------------------------------------------
// JsonDiff — generic recursive JSON differ
// ---------------------------------------------------------------------------

/// Recursively compares two arbitrary `serde_json::Value` trees and returns a
/// flat list of [`JsonChange`]s using JSON Pointer notation for paths.
pub struct JsonDiff;

impl JsonDiff {
    pub fn diff(old: &serde_json::Value, new: &serde_json::Value) -> Vec<JsonChange> {
        let mut changes = Vec::new();
        Self::diff_recursive(old, new, String::new(), &mut changes);
        changes
    }

    fn diff_recursive(
        old: &serde_json::Value,
        new: &serde_json::Value,
        path: String,
        changes: &mut Vec<JsonChange>,
    ) {
        use serde_json::Value;

        match (old, new) {
            (Value::Object(old_map), Value::Object(new_map)) => {
                // Collect all keys from both sides.
                let mut all_keys: Vec<&String> = old_map.keys().chain(new_map.keys()).collect();
                all_keys.sort();
                all_keys.dedup();

                for key in all_keys {
                    let child_path = format!("{}/{}", path, Self::escape_pointer(key));
                    match (old_map.get(key), new_map.get(key)) {
                        (Some(ov), Some(nv)) => {
                            Self::diff_recursive(ov, nv, child_path, changes);
                        }
                        (Some(ov), None) => {
                            changes.push(JsonChange {
                                path: child_path,
                                old_value: Some(ov.clone()),
                                new_value: None,
                                change_type: DiffChangeType::Removed,
                            });
                        }
                        (None, Some(nv)) => {
                            changes.push(JsonChange {
                                path: child_path,
                                old_value: None,
                                new_value: Some(nv.clone()),
                                change_type: DiffChangeType::Added,
                            });
                        }
                        (None, None) => unreachable!(),
                    }
                }
            }
            (Value::Array(old_arr), Value::Array(new_arr)) => {
                let max_len = old_arr.len().max(new_arr.len());
                for i in 0..max_len {
                    let child_path = format!("{}/{}", path, i);
                    match (old_arr.get(i), new_arr.get(i)) {
                        (Some(ov), Some(nv)) => {
                            Self::diff_recursive(ov, nv, child_path, changes);
                        }
                        (Some(ov), None) => {
                            changes.push(JsonChange {
                                path: child_path,
                                old_value: Some(ov.clone()),
                                new_value: None,
                                change_type: DiffChangeType::Removed,
                            });
                        }
                        (None, Some(nv)) => {
                            changes.push(JsonChange {
                                path: child_path,
                                old_value: None,
                                new_value: Some(nv.clone()),
                                change_type: DiffChangeType::Added,
                            });
                        }
                        (None, None) => unreachable!(),
                    }
                }
            }
            _ => {
                if old != new {
                    changes.push(JsonChange {
                        path: if path.is_empty() {
                            "/".to_string()
                        } else {
                            path
                        },
                        old_value: Some(old.clone()),
                        new_value: Some(new.clone()),
                        change_type: DiffChangeType::Modified,
                    });
                }
            }
        }
    }

    /// Escape `~` and `/` per RFC 6901 JSON Pointer.
    fn escape_pointer(segment: &str) -> String {
        segment.replace('~', "~0").replace('/', "~1")
    }
}

// ---------------------------------------------------------------------------
// PlanDiff — main diff engine
// ---------------------------------------------------------------------------

/// Main engine for computing deployment-plan diffs.
///
/// Plans are represented as `serde_json::Value` objects. Steps inside a plan
/// are matched by an `"id"` or `"step_id"` field.
pub struct PlanDiff;

impl PlanDiff {
    /// Diff two complete plan JSON documents.
    ///
    /// Expects each plan to have a top-level `"steps"` array. Falls back to
    /// comparing the root values directly when the key is absent.
    pub fn diff_plans(
        old: &serde_json::Value,
        new: &serde_json::Value,
    ) -> PlanDiffResult {
        let empty_arr = serde_json::Value::Array(Vec::new());

        let old_steps_val = old.get("steps").unwrap_or(&empty_arr);
        let new_steps_val = new.get("steps").unwrap_or(&empty_arr);

        let old_steps: Vec<serde_json::Value> = match old_steps_val {
            serde_json::Value::Array(a) => a.clone(),
            _ => Vec::new(),
        };
        let new_steps: Vec<serde_json::Value> = match new_steps_val {
            serde_json::Value::Array(a) => a.clone(),
            _ => Vec::new(),
        };

        let step_diffs = Self::diff_step_lists(&old_steps, &new_steps);

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        for sd in step_diffs {
            match sd.change_type {
                DiffChangeType::Added => added.push(sd),
                DiffChangeType::Removed => removed.push(sd),
                DiffChangeType::Modified => modified.push(sd),
                DiffChangeType::Unchanged => {}
            }
        }

        let safety_impact = Self::compute_safety_impact(old, new, &added, &removed, &modified);
        let summary = Self::build_summary(&added, &removed, &modified);

        PlanDiffResult {
            added_steps: added,
            removed_steps: removed,
            modified_steps: modified,
            safety_impact,
            summary,
        }
    }

    /// Compare two flat lists of step JSON values.
    ///
    /// Steps are matched by their `"id"` or `"step_id"` field. Steps that
    /// exist only in the old list are marked `Removed`, only in the new list
    /// `Added`, and in both lists are compared for equality (`Modified` vs
    /// `Unchanged`).
    pub fn diff_step_lists(
        old_steps: &[serde_json::Value],
        new_steps: &[serde_json::Value],
    ) -> Vec<StepDiff> {
        let extract_id = |v: &serde_json::Value| -> String {
            v.get("id")
                .or_else(|| v.get("step_id"))
                .and_then(|id| id.as_str())
                .unwrap_or("")
                .to_string()
        };

        let mut old_map: indexmap::IndexMap<String, &serde_json::Value> = indexmap::IndexMap::new();
        for step in old_steps {
            let id = extract_id(step);
            if !id.is_empty() {
                old_map.insert(id, step);
            }
        }

        let mut new_map: indexmap::IndexMap<String, &serde_json::Value> = indexmap::IndexMap::new();
        for step in new_steps {
            let id = extract_id(step);
            if !id.is_empty() {
                new_map.insert(id, step);
            }
        }

        let mut results = Vec::new();

        // Walk old steps – detect removed and modified/unchanged.
        for (id, old_val) in &old_map {
            if let Some(new_val) = new_map.get(id) {
                let old_str = serde_json::to_string(old_val).unwrap_or_default();
                let new_str = serde_json::to_string(new_val).unwrap_or_default();
                if old_str == new_str {
                    results.push(StepDiff::unchanged(id.clone(), old_str));
                } else {
                    results.push(StepDiff::modified(id.clone(), old_str, new_str));
                }
            } else {
                let old_str = serde_json::to_string(old_val).unwrap_or_default();
                results.push(StepDiff::removed(id.clone(), old_str));
            }
        }

        // Walk new steps – detect added.
        for (id, new_val) in &new_map {
            if !old_map.contains_key(id) {
                let new_str = serde_json::to_string(new_val).unwrap_or_default();
                results.push(StepDiff::added(id.clone(), new_str));
            }
        }

        results
    }

    // -- private helpers -----------------------------------------------------

    fn compute_safety_impact(
        old: &serde_json::Value,
        new: &serde_json::Value,
        added: &[StepDiff],
        removed: &[StepDiff],
        modified: &[StepDiff],
    ) -> SafetyImpact {
        let count_pnr = |plan: &serde_json::Value| -> i64 {
            plan.get("pnr_states")
                .and_then(|v| v.as_array())
                .map(|a| a.len() as i64)
                .unwrap_or(0)
        };

        let old_pnr = count_pnr(old);
        let new_pnr = count_pnr(new);
        let pnr_delta = new_pnr - old_pnr;

        let read_envelope = |plan: &serde_json::Value| -> f64 {
            plan.get("envelope_size")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };

        let old_env = read_envelope(old);
        let new_env = read_envelope(new);
        let envelope_change = new_env - old_env;

        // Simple heuristic: each added step adds a small risk, each removed
        // step reduces risk, modified steps add minor risk.
        let base_risk: f64 =
            (added.len() as f64) * 0.1 - (removed.len() as f64) * 0.05 + (modified.len() as f64) * 0.05;

        // If the plan carries an explicit risk score, prefer its delta.
        let explicit_risk = |plan: &serde_json::Value| -> Option<f64> {
            plan.get("risk_score").and_then(|v| v.as_f64())
        };

        let risk_delta = match (explicit_risk(old), explicit_risk(new)) {
            (Some(or), Some(nr)) => nr - or,
            _ => base_risk,
        };

        SafetyImpact {
            new_pnr_count: pnr_delta,
            envelope_change,
            risk_delta,
        }
    }

    fn build_summary(
        added: &[StepDiff],
        removed: &[StepDiff],
        modified: &[StepDiff],
    ) -> String {
        if added.is_empty() && removed.is_empty() && modified.is_empty() {
            return "No changes detected".to_string();
        }

        let mut parts = Vec::new();
        if !added.is_empty() {
            parts.push(format!("{} added", added.len()));
        }
        if !removed.is_empty() {
            parts.push(format!("{} removed", removed.len()));
        }
        if !modified.is_empty() {
            parts.push(format!("{} modified", modified.len()));
        }
        parts.join(", ")
    }
}

// ---------------------------------------------------------------------------
// DiffFormatter
// ---------------------------------------------------------------------------

/// Renders a [`PlanDiffResult`] in various human- and machine-readable formats.
pub struct DiffFormatter;

impl DiffFormatter {
    /// Plain-text format with `+` / `-` / `~` markers for each step change.
    pub fn format_text(diff: &PlanDiffResult) -> String {
        let mut out = String::new();

        out.push_str("Plan Diff\n");
        out.push_str(&"=".repeat(40));
        out.push('\n');

        if !diff.added_steps.is_empty() {
            out.push_str("\nAdded steps:\n");
            for s in &diff.added_steps {
                out.push_str(&format!(
                    "  + [{}] {}\n",
                    s.step_id,
                    s.new_value.as_deref().unwrap_or("(empty)")
                ));
            }
        }

        if !diff.removed_steps.is_empty() {
            out.push_str("\nRemoved steps:\n");
            for s in &diff.removed_steps {
                out.push_str(&format!(
                    "  - [{}] {}\n",
                    s.step_id,
                    s.old_value.as_deref().unwrap_or("(empty)")
                ));
            }
        }

        if !diff.modified_steps.is_empty() {
            out.push_str("\nModified steps:\n");
            for s in &diff.modified_steps {
                out.push_str(&format!("  ~ [{}]\n", s.step_id));
                out.push_str(&format!(
                    "    - {}\n",
                    s.old_value.as_deref().unwrap_or("(empty)")
                ));
                out.push_str(&format!(
                    "    + {}\n",
                    s.new_value.as_deref().unwrap_or("(empty)")
                ));
            }
        }

        out.push_str(&format!("\nSafety: severity={}, risk_delta={:.2}, pnr_delta={}\n",
            diff.safety_impact.severity(),
            diff.safety_impact.risk_delta,
            diff.safety_impact.new_pnr_count,
        ));

        out.push_str(&format!("Summary: {}\n", diff.summary));

        out
    }

    /// Serialize the diff result to a JSON string.
    pub fn format_json(diff: &PlanDiffResult) -> String {
        serde_json::to_string_pretty(diff).unwrap_or_else(|_| "{}".to_string())
    }

    /// One-line summary such as `"3 added, 1 removed, 2 modified"`.
    pub fn format_summary(diff: &PlanDiffResult) -> String {
        if !diff.has_changes() {
            return "No changes".to_string();
        }

        let mut parts = Vec::new();
        if !diff.added_steps.is_empty() {
            parts.push(format!("{} added", diff.added_steps.len()));
        }
        if !diff.removed_steps.is_empty() {
            parts.push(format!("{} removed", diff.removed_steps.len()));
        }
        if !diff.modified_steps.is_empty() {
            parts.push(format!("{} modified", diff.modified_steps.len()));
        }
        parts.join(", ")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- DiffChangeType ---------------------------------------------------

    #[test]
    fn test_diff_change_type_display() {
        assert_eq!(DiffChangeType::Added.to_string(), "added");
        assert_eq!(DiffChangeType::Removed.to_string(), "removed");
        assert_eq!(DiffChangeType::Modified.to_string(), "modified");
        assert_eq!(DiffChangeType::Unchanged.to_string(), "unchanged");
    }

    // ---- StepDiff constructors --------------------------------------------

    #[test]
    fn test_step_diff_added() {
        let sd = StepDiff::added("s1", "payload");
        assert_eq!(sd.change_type, DiffChangeType::Added);
        assert!(sd.old_value.is_none());
        assert_eq!(sd.new_value.as_deref(), Some("payload"));
    }

    #[test]
    fn test_step_diff_removed() {
        let sd = StepDiff::removed("s2", "old_payload");
        assert_eq!(sd.change_type, DiffChangeType::Removed);
        assert_eq!(sd.old_value.as_deref(), Some("old_payload"));
        assert!(sd.new_value.is_none());
    }

    #[test]
    fn test_step_diff_modified() {
        let sd = StepDiff::modified("s3", "before", "after");
        assert_eq!(sd.change_type, DiffChangeType::Modified);
        assert_eq!(sd.old_value.as_deref(), Some("before"));
        assert_eq!(sd.new_value.as_deref(), Some("after"));
    }

    #[test]
    fn test_step_diff_unchanged() {
        let sd = StepDiff::unchanged("s4", "same");
        assert_eq!(sd.change_type, DiffChangeType::Unchanged);
        assert_eq!(sd.old_value, sd.new_value);
    }

    // ---- SafetyImpact -----------------------------------------------------

    #[test]
    fn test_safety_impact_none_is_safe() {
        let si = SafetyImpact::none();
        assert!(si.is_safe_change());
        assert_eq!(si.severity(), "none");
    }

    #[test]
    fn test_safety_impact_safe_negative_risk() {
        let si = SafetyImpact {
            new_pnr_count: -1,
            envelope_change: 0.0,
            risk_delta: -0.2,
        };
        assert!(si.is_safe_change());
        assert_eq!(si.severity(), "low");
    }

    #[test]
    fn test_safety_impact_unsafe_positive_risk() {
        let si = SafetyImpact {
            new_pnr_count: 2,
            envelope_change: 0.5,
            risk_delta: 0.8,
        };
        assert!(!si.is_safe_change());
        assert_eq!(si.severity(), "high");
    }

    #[test]
    fn test_safety_impact_medium_severity() {
        let si = SafetyImpact {
            new_pnr_count: 0,
            envelope_change: 0.0,
            risk_delta: 0.5,
        };
        assert!(!si.is_safe_change());
        assert_eq!(si.severity(), "medium");
    }

    #[test]
    fn test_safety_impact_unsafe_due_to_pnr_only() {
        let si = SafetyImpact {
            new_pnr_count: 1,
            envelope_change: 0.0,
            risk_delta: 0.0,
        };
        assert!(!si.is_safe_change());
        assert_eq!(si.severity(), "none");
    }

    // ---- PlanDiffResult ---------------------------------------------------

    #[test]
    fn test_plan_diff_result_empty() {
        let r = PlanDiffResult::empty();
        assert_eq!(r.total_changes(), 0);
        assert!(!r.has_changes());
    }

    #[test]
    fn test_plan_diff_result_counts() {
        let r = PlanDiffResult {
            added_steps: vec![StepDiff::added("a", "x")],
            removed_steps: vec![StepDiff::removed("b", "y"), StepDiff::removed("c", "z")],
            modified_steps: vec![StepDiff::modified("d", "o", "n")],
            safety_impact: SafetyImpact::none(),
            summary: String::new(),
        };
        assert_eq!(r.total_changes(), 4);
        assert!(r.has_changes());
    }

    // ---- DiffImpact & VersionDiff -----------------------------------------

    #[test]
    fn test_diff_impact_display() {
        assert_eq!(DiffImpact::Breaking.to_string(), "breaking");
        assert_eq!(DiffImpact::NonBreaking.to_string(), "non-breaking");
        assert_eq!(DiffImpact::Neutral.to_string(), "neutral");
        assert_eq!(DiffImpact::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_version_diff_infer_breaking() {
        assert_eq!(VersionDiff::infer_impact("1.0.0", "2.0.0"), DiffImpact::Breaking);
        assert_eq!(VersionDiff::infer_impact("v1.2.3", "v2.0.0"), DiffImpact::Breaking);
    }

    #[test]
    fn test_version_diff_infer_nonbreaking() {
        assert_eq!(VersionDiff::infer_impact("1.0.0", "1.1.0"), DiffImpact::NonBreaking);
        assert_eq!(VersionDiff::infer_impact("1.0.0", "1.0.1"), DiffImpact::NonBreaking);
    }

    #[test]
    fn test_version_diff_infer_neutral() {
        assert_eq!(VersionDiff::infer_impact("1.2.3", "1.2.3"), DiffImpact::Neutral);
    }

    #[test]
    fn test_version_diff_infer_unknown() {
        assert_eq!(VersionDiff::infer_impact("latest", "canary"), DiffImpact::Unknown);
    }

    #[test]
    fn test_version_diff_new() {
        let vd = VersionDiff::new("svc-a", "1.0.0", "2.0.0", DiffImpact::Breaking);
        assert_eq!(vd.service, "svc-a");
        assert_eq!(vd.impact, DiffImpact::Breaking);
    }

    // ---- SafetyChange -----------------------------------------------------

    #[test]
    fn test_safety_change_degradation() {
        let sc = SafetyChange::new("went bad", "safe", "pnr", vec!["s1".into()]);
        assert!(sc.is_degradation());
    }

    #[test]
    fn test_safety_change_not_degradation_improvement() {
        let sc = SafetyChange::new("got better", "pnr", "safe", vec![]);
        assert!(!sc.is_degradation());
    }

    #[test]
    fn test_safety_change_not_degradation_same() {
        let sc = SafetyChange::new("no change", "safe", "safe", vec![]);
        assert!(!sc.is_degradation());
    }

    #[test]
    fn test_safety_change_degradation_case_insensitive() {
        let sc = SafetyChange::new("oops", "Safe", "PNR", vec!["x".into()]);
        assert!(sc.is_degradation());
    }

    #[test]
    fn test_safety_change_degradation_green_to_red() {
        let sc = SafetyChange::new("alert", "green", "red", vec![]);
        assert!(sc.is_degradation());
    }

    // ---- PlanDiff ---------------------------------------------------------

    #[test]
    fn test_diff_identical_plans() {
        let plan = json!({
            "steps": [
                {"id": "s1", "action": "deploy"},
                {"id": "s2", "action": "verify"}
            ]
        });
        let result = PlanDiff::diff_plans(&plan, &plan);
        assert!(!result.has_changes());
        assert_eq!(result.summary, "No changes detected");
    }

    #[test]
    fn test_diff_added_step() {
        let old = json!({ "steps": [{"id": "s1", "action": "deploy"}] });
        let new = json!({ "steps": [{"id": "s1", "action": "deploy"}, {"id": "s2", "action": "verify"}] });
        let result = PlanDiff::diff_plans(&old, &new);
        assert_eq!(result.added_steps.len(), 1);
        assert_eq!(result.added_steps[0].step_id, "s2");
        assert!(result.removed_steps.is_empty());
    }

    #[test]
    fn test_diff_removed_step() {
        let old = json!({ "steps": [{"id": "s1"}, {"id": "s2"}] });
        let new = json!({ "steps": [{"id": "s1"}] });
        let result = PlanDiff::diff_plans(&old, &new);
        assert_eq!(result.removed_steps.len(), 1);
        assert_eq!(result.removed_steps[0].step_id, "s2");
    }

    #[test]
    fn test_diff_modified_step() {
        let old = json!({ "steps": [{"id": "s1", "action": "deploy"}] });
        let new = json!({ "steps": [{"id": "s1", "action": "rollback"}] });
        let result = PlanDiff::diff_plans(&old, &new);
        assert_eq!(result.modified_steps.len(), 1);
        assert_eq!(result.modified_steps[0].step_id, "s1");
    }

    #[test]
    fn test_diff_empty_plans() {
        let old = json!({ "steps": [] });
        let new = json!({ "steps": [] });
        let result = PlanDiff::diff_plans(&old, &new);
        assert!(!result.has_changes());
    }

    #[test]
    fn test_diff_completely_different_plans() {
        let old = json!({ "steps": [{"id": "a"}, {"id": "b"}] });
        let new = json!({ "steps": [{"id": "c"}, {"id": "d"}] });
        let result = PlanDiff::diff_plans(&old, &new);
        assert_eq!(result.removed_steps.len(), 2);
        assert_eq!(result.added_steps.len(), 2);
        assert!(result.modified_steps.is_empty());
    }

    #[test]
    fn test_diff_step_id_field() {
        let old = json!({ "steps": [{"step_id": "x1", "v": 1}] });
        let new = json!({ "steps": [{"step_id": "x1", "v": 2}] });
        let result = PlanDiff::diff_plans(&old, &new);
        assert_eq!(result.modified_steps.len(), 1);
        assert_eq!(result.modified_steps[0].step_id, "x1");
    }

    #[test]
    fn test_diff_no_steps_key() {
        let old = json!({"name": "plan-a"});
        let new = json!({"name": "plan-b"});
        let result = PlanDiff::diff_plans(&old, &new);
        assert!(!result.has_changes());
    }

    #[test]
    fn test_diff_plans_with_risk_scores() {
        let old = json!({ "steps": [{"id": "s1"}], "risk_score": 0.2 });
        let new = json!({ "steps": [{"id": "s1"}, {"id": "s2"}], "risk_score": 0.5 });
        let result = PlanDiff::diff_plans(&old, &new);
        assert!((result.safety_impact.risk_delta - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_diff_plans_pnr_delta() {
        let old = json!({ "steps": [], "pnr_states": ["p1"] });
        let new = json!({ "steps": [], "pnr_states": ["p1", "p2", "p3"] });
        let result = PlanDiff::diff_plans(&old, &new);
        assert_eq!(result.safety_impact.new_pnr_count, 2);
    }

    // ---- diff_step_lists standalone ---------------------------------------

    #[test]
    fn test_diff_step_lists_mixed() {
        let old = vec![
            json!({"id": "a", "v": 1}),
            json!({"id": "b", "v": 2}),
        ];
        let new = vec![
            json!({"id": "b", "v": 2}),
            json!({"id": "c", "v": 3}),
        ];
        let diffs = PlanDiff::diff_step_lists(&old, &new);
        let removed: Vec<_> = diffs.iter().filter(|d| d.change_type == DiffChangeType::Removed).collect();
        let added: Vec<_> = diffs.iter().filter(|d| d.change_type == DiffChangeType::Added).collect();
        let unchanged: Vec<_> = diffs.iter().filter(|d| d.change_type == DiffChangeType::Unchanged).collect();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].step_id, "a");
        assert_eq!(added.len(), 1);
        assert_eq!(added[0].step_id, "c");
        assert_eq!(unchanged.len(), 1);
        assert_eq!(unchanged[0].step_id, "b");
    }

    // ---- DiffFormatter ----------------------------------------------------

    #[test]
    fn test_format_summary_no_changes() {
        let r = PlanDiffResult::empty();
        assert_eq!(DiffFormatter::format_summary(&r), "No changes");
    }

    #[test]
    fn test_format_summary_mixed() {
        let r = PlanDiffResult {
            added_steps: vec![StepDiff::added("a", "x"), StepDiff::added("b", "y"), StepDiff::added("c", "z")],
            removed_steps: vec![StepDiff::removed("d", "w")],
            modified_steps: vec![StepDiff::modified("e", "o", "n"), StepDiff::modified("f", "o2", "n2")],
            safety_impact: SafetyImpact::none(),
            summary: String::new(),
        };
        assert_eq!(DiffFormatter::format_summary(&r), "3 added, 1 removed, 2 modified");
    }

    #[test]
    fn test_format_text_contains_markers() {
        let r = PlanDiffResult {
            added_steps: vec![StepDiff::added("a1", "new-step")],
            removed_steps: vec![StepDiff::removed("r1", "old-step")],
            modified_steps: vec![StepDiff::modified("m1", "before", "after")],
            safety_impact: SafetyImpact::none(),
            summary: "1 added, 1 removed, 1 modified".into(),
        };
        let text = DiffFormatter::format_text(&r);
        assert!(text.contains("+ [a1]"));
        assert!(text.contains("- [r1]"));
        assert!(text.contains("~ [m1]"));
        assert!(text.contains("Plan Diff"));
    }

    #[test]
    fn test_format_json_roundtrip() {
        let r = PlanDiffResult {
            added_steps: vec![StepDiff::added("j1", "val")],
            removed_steps: vec![],
            modified_steps: vec![],
            safety_impact: SafetyImpact::none(),
            summary: "1 added".into(),
        };
        let json_str = DiffFormatter::format_json(&r);
        let parsed: PlanDiffResult = serde_json::from_str(&json_str).expect("valid json");
        assert_eq!(parsed.added_steps.len(), 1);
        assert_eq!(parsed.added_steps[0].step_id, "j1");
    }

    // ---- JsonDiff ---------------------------------------------------------

    #[test]
    fn test_json_diff_identical() {
        let v = json!({"a": 1, "b": [2, 3]});
        let changes = JsonDiff::diff(&v, &v);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_json_diff_scalar_change() {
        let old = json!({"x": 10});
        let new = json!({"x": 20});
        let changes = JsonDiff::diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/x");
        assert_eq!(changes[0].change_type, DiffChangeType::Modified);
    }

    #[test]
    fn test_json_diff_added_key() {
        let old = json!({"a": 1});
        let new = json!({"a": 1, "b": 2});
        let changes = JsonDiff::diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/b");
        assert_eq!(changes[0].change_type, DiffChangeType::Added);
    }

    #[test]
    fn test_json_diff_removed_key() {
        let old = json!({"a": 1, "b": 2});
        let new = json!({"a": 1});
        let changes = JsonDiff::diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/b");
        assert_eq!(changes[0].change_type, DiffChangeType::Removed);
    }

    #[test]
    fn test_json_diff_nested_object() {
        let old = json!({"outer": {"inner": 1}});
        let new = json!({"outer": {"inner": 2}});
        let changes = JsonDiff::diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/outer/inner");
        assert_eq!(changes[0].change_type, DiffChangeType::Modified);
    }

    #[test]
    fn test_json_diff_array_element_changed() {
        let old = json!({"arr": [1, 2, 3]});
        let new = json!({"arr": [1, 99, 3]});
        let changes = JsonDiff::diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/arr/1");
    }

    #[test]
    fn test_json_diff_array_length_increase() {
        let old = json!({"arr": [1]});
        let new = json!({"arr": [1, 2, 3]});
        let changes = JsonDiff::diff(&old, &new);
        let added: Vec<_> = changes.iter().filter(|c| c.change_type == DiffChangeType::Added).collect();
        assert_eq!(added.len(), 2);
    }

    #[test]
    fn test_json_diff_array_length_decrease() {
        let old = json!({"arr": [1, 2, 3]});
        let new = json!({"arr": [1]});
        let changes = JsonDiff::diff(&old, &new);
        let removed: Vec<_> = changes.iter().filter(|c| c.change_type == DiffChangeType::Removed).collect();
        assert_eq!(removed.len(), 2);
    }

    #[test]
    fn test_json_diff_type_change() {
        let old = json!({"v": 42});
        let new = json!({"v": "forty-two"});
        let changes = JsonDiff::diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].change_type, DiffChangeType::Modified);
    }

    #[test]
    fn test_json_diff_deeply_nested() {
        let old = json!({"a": {"b": {"c": {"d": 1}}}});
        let new = json!({"a": {"b": {"c": {"d": 2}}}});
        let changes = JsonDiff::diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/a/b/c/d");
    }

    #[test]
    fn test_json_diff_empty_objects() {
        let changes = JsonDiff::diff(&json!({}), &json!({}));
        assert!(changes.is_empty());
    }

    #[test]
    fn test_json_diff_root_scalar() {
        let changes = JsonDiff::diff(&json!(1), &json!(2));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path, "/");
        assert_eq!(changes[0].change_type, DiffChangeType::Modified);
    }

    #[test]
    fn test_json_diff_pointer_escaping() {
        let old = json!({"a/b": 1, "c~d": 2});
        let new = json!({"a/b": 10, "c~d": 20});
        let changes = JsonDiff::diff(&old, &new);
        assert_eq!(changes.len(), 2);
        let paths: Vec<&str> = changes.iter().map(|c| c.path.as_str()).collect();
        assert!(paths.contains(&"/a~1b"));
        assert!(paths.contains(&"/c~0d"));
    }

    // ---- Serialization round-trips ----------------------------------------

    #[test]
    fn test_serde_roundtrip_step_diff() {
        let sd = StepDiff::modified("rt", "old", "new");
        let json_str = serde_json::to_string(&sd).unwrap();
        let back: StepDiff = serde_json::from_str(&json_str).unwrap();
        assert_eq!(back, sd);
    }

    #[test]
    fn test_serde_roundtrip_safety_impact() {
        let si = SafetyImpact {
            new_pnr_count: 3,
            envelope_change: -0.5,
            risk_delta: 0.42,
        };
        let json_str = serde_json::to_string(&si).unwrap();
        let back: SafetyImpact = serde_json::from_str(&json_str).unwrap();
        assert_eq!(back, si);
    }

    #[test]
    fn test_serde_roundtrip_version_diff() {
        let vd = VersionDiff::new("svc", "1.0.0", "2.0.0", DiffImpact::Breaking);
        let json_str = serde_json::to_string(&vd).unwrap();
        let back: VersionDiff = serde_json::from_str(&json_str).unwrap();
        assert_eq!(back, vd);
    }

    #[test]
    fn test_serde_roundtrip_safety_change() {
        let sc = SafetyChange::new("desc", "safe", "pnr", vec!["s1".into()]);
        let json_str = serde_json::to_string(&sc).unwrap();
        let back: SafetyChange = serde_json::from_str(&json_str).unwrap();
        assert_eq!(back, sc);
    }
}

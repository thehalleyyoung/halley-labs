//! Generate concrete configuration diffs from repair plans.
//!
//! The [`ConfigDiffGenerator`] takes a [`RepairPlan`] and the original
//! configuration files (as raw strings keyed by file path) and produces
//! unified diffs, YAML strategic-merge patches, and coloured output.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{RepairAction, RepairActionType, RepairPlan};

// ---------------------------------------------------------------------------
// Diff types
// ---------------------------------------------------------------------------

/// A line inside a diff hunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffLine {
    Context(String),
    Added(String),
    Removed(String),
}

impl std::fmt::Display for DiffLine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffLine::Context(s) => write!(f, " {}", s),
            DiffLine::Added(s) => write!(f, "+{}", s),
            DiffLine::Removed(s) => write!(f, "-{}", s),
        }
    }
}

/// A contiguous hunk inside a file diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffHunk {
    pub header: String,
    pub lines: Vec<DiffLine>,
}

impl DiffHunk {
    /// Format the hunk as a unified-diff fragment.
    pub fn to_unified(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.header);
        out.push('\n');
        for line in &self.lines {
            out.push_str(&line.to_string());
            out.push('\n');
        }
        out
    }
}

/// Changes in a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChange {
    pub file_path: String,
    pub hunks: Vec<DiffHunk>,
}

/// A complete diff across one or more files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigDiff {
    pub changes: Vec<FileChange>,
    pub summary: String,
}

impl ConfigDiff {
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Number of lines added across all hunks.
    pub fn lines_added(&self) -> usize {
        self.changes
            .iter()
            .flat_map(|fc| fc.hunks.iter())
            .flat_map(|h| h.lines.iter())
            .filter(|l| matches!(l, DiffLine::Added(_)))
            .count()
    }

    /// Number of lines removed across all hunks.
    pub fn lines_removed(&self) -> usize {
        self.changes
            .iter()
            .flat_map(|fc| fc.hunks.iter())
            .flat_map(|h| h.lines.iter())
            .filter(|l| matches!(l, DiffLine::Removed(_)))
            .count()
    }
}

// ---------------------------------------------------------------------------
// LCS-based line diff
// ---------------------------------------------------------------------------

/// Compute a simple line-level diff between `old_lines` and `new_lines`
/// using the Longest Common Subsequence algorithm.
pub fn compute_line_diff(old_lines: &[&str], new_lines: &[&str]) -> Vec<DiffLine> {
    let n = old_lines.len();
    let m = new_lines.len();

    // Build LCS table.
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            if old_lines[i - 1] == new_lines[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Back-trace to produce diff.
    let mut result: Vec<DiffLine> = Vec::new();
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && old_lines[i - 1] == new_lines[j - 1] {
            result.push(DiffLine::Context(old_lines[i - 1].to_string()));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
            result.push(DiffLine::Added(new_lines[j - 1].to_string()));
            j -= 1;
        } else {
            result.push(DiffLine::Removed(old_lines[i - 1].to_string()));
            i -= 1;
        }
    }

    result.reverse();
    result
}

// ---------------------------------------------------------------------------
// ConfigDiffGenerator
// ---------------------------------------------------------------------------

/// Generates diffs, YAML patches, and coloured output from repair plans.
#[derive(Debug, Default)]
pub struct ConfigDiffGenerator {
    _private: (),
}

impl ConfigDiffGenerator {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Generate a [`ConfigDiff`] by applying the repair plan to the original
    /// config files.
    ///
    /// `original_configs` maps *file path* → *YAML / JSON content*.
    pub fn generate_diff(
        &self,
        repair: &RepairPlan,
        original_configs: &HashMap<String, String>,
    ) -> ConfigDiff {
        let mut changes: Vec<FileChange> = Vec::new();

        // Group actions by the file they affect.  We use the service name
        // as a proxy for file path: `<service>.yaml`.
        let mut actions_by_file: HashMap<String, Vec<&RepairAction>> = HashMap::new();
        for action in &repair.actions {
            let file = if let Some((src, _)) = &action.edge {
                format!("{}.yaml", src)
            } else {
                format!("{}.yaml", action.service)
            };
            actions_by_file.entry(file).or_default().push(action);
        }

        for (file_path, actions) in &actions_by_file {
            let original = original_configs.get(file_path).cloned().unwrap_or_default();
            let mut modified = original.clone();
            for action in actions {
                modified = self.apply_repair_to_yaml(&modified, action);
            }

            let old_lines: Vec<&str> = original.lines().collect();
            let new_lines: Vec<&str> = modified.lines().collect();
            let diff_lines = compute_line_diff(&old_lines, &new_lines);

            if diff_lines.iter().any(|l| !matches!(l, DiffLine::Context(_))) {
                let header = format!("@@ file: {} @@", file_path);
                changes.push(FileChange {
                    file_path: file_path.clone(),
                    hunks: vec![DiffHunk {
                        header,
                        lines: diff_lines,
                    }],
                });
            }
        }

        let summary = if changes.is_empty() {
            "No configuration changes required.".to_string()
        } else {
            format!(
                "{} file(s) changed with {} action(s)",
                changes.len(),
                repair.actions.len()
            )
        };

        ConfigDiff { changes, summary }
    }

    /// Create a kubectl-compatible strategic merge patch in YAML format.
    pub fn generate_yaml_patch(&self, repair: &RepairPlan) -> String {
        let mut lines: Vec<String> = Vec::new();
        lines.push("apiVersion: networking.istio.io/v1alpha3".to_string());
        lines.push("kind: DestinationRule".to_string());
        lines.push("metadata:".to_string());
        lines.push("  name: cascade-repair-patch".to_string());
        lines.push("spec:".to_string());
        lines.push("  trafficPolicy:".to_string());

        for action in &repair.actions {
            match &action.action_type {
                RepairActionType::ReduceRetries { to, .. } => {
                    let edge_name = action
                        .edge
                        .as_ref()
                        .map(|(s, t)| format!("{}-{}", s, t))
                        .unwrap_or_else(|| action.service.clone());
                    lines.push(format!("    # edge: {}", edge_name));
                    lines.push("    retries:".to_string());
                    lines.push(format!("      attempts: {}", to));
                }
                RepairActionType::AdjustTimeout { to_ms, .. } => {
                    let edge_name = action
                        .edge
                        .as_ref()
                        .map(|(s, t)| format!("{}-{}", s, t))
                        .unwrap_or_else(|| action.service.clone());
                    lines.push(format!("    # edge: {}", edge_name));
                    lines.push("    connectionPool:".to_string());
                    lines.push("      tcp:".to_string());
                    lines.push(format!("        connectTimeout: {}ms", to_ms));
                }
                _ => {}
            }
        }

        lines.join("\n") + "\n"
    }

    /// Standard unified diff format.
    pub fn generate_unified_diff(&self, original: &str, modified: &str) -> String {
        let old_lines: Vec<&str> = original.lines().collect();
        let new_lines: Vec<&str> = modified.lines().collect();
        let diff = compute_line_diff(&old_lines, &new_lines);

        let mut out = String::new();
        out.push_str("--- a/original\n");
        out.push_str("+++ b/modified\n");

        // Build hunks from contiguous non-context regions with surrounding
        // context lines (up to 3).
        let hunks = build_hunks_from_diff(&diff, 3);
        for hunk in hunks {
            out.push_str(&hunk.to_unified());
        }
        out
    }

    /// Apply a single repair action to a YAML string, returning the modified
    /// YAML.
    pub fn apply_repair_to_yaml(&self, yaml_content: &str, action: &RepairAction) -> String {
        let lines: Vec<&str> = yaml_content.lines().collect();
        let mut result: Vec<String> = Vec::new();

        let (field, new_val) = match &action.action_type {
            RepairActionType::ReduceRetries { to, .. } => ("attempts", format!("{}", to)),
            RepairActionType::AdjustTimeout { to_ms, .. } => {
                ("connectTimeout", format!("{}ms", to_ms))
            }
            RepairActionType::AddCircuitBreaker { threshold } => {
                ("consecutiveErrors", format!("{}", threshold))
            }
            RepairActionType::AddRateLimit { rps } => ("requestsPerSecond", format!("{:.1}", rps)),
            RepairActionType::IncreaseCapacity { to, .. } => ("capacity", format!("{:.1}", to)),
        };

        let edge_label = action
            .edge
            .as_ref()
            .map(|(s, t)| format!("{}->{}", s, t))
            .unwrap_or_default();

        let mut found = false;
        let mut in_edge_section = edge_label.is_empty(); // if no edge, always match

        for line in &lines {
            // Detect edge section markers (comments like `# edge: A->B`).
            if line.contains(&edge_label) && !edge_label.is_empty() {
                in_edge_section = true;
            }

            if in_edge_section && line.trim_start().starts_with(&format!("{}:", field)) {
                let indent = line.len() - line.trim_start().len();
                let spaces: String = " ".repeat(indent);
                result.push(format!("{}{}: {}", spaces, field, new_val));
                found = true;
                in_edge_section = false;
            } else {
                result.push(line.to_string());
            }
        }

        // If the field was not found, append it.
        if !found {
            result.push(format!("# repair for {}", action.description));
            result.push(format!("{}: {}", field, new_val));
        }

        result.join("\n")
    }

    /// Format a diff with colour markers (`[+]` / `[-]`).
    pub fn format_diff_colored(&self, diff: &ConfigDiff) -> String {
        let mut out = String::new();
        for fc in &diff.changes {
            out.push_str(&format!("=== {} ===\n", fc.file_path));
            for hunk in &fc.hunks {
                out.push_str(&hunk.header);
                out.push('\n');
                for line in &hunk.lines {
                    match line {
                        DiffLine::Context(s) => {
                            out.push_str(&format!("  {}\n", s));
                        }
                        DiffLine::Added(s) => {
                            out.push_str(&format!("[+] {}\n", s));
                        }
                        DiffLine::Removed(s) => {
                            out.push_str(&format!("[-] {}\n", s));
                        }
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Split a flat `Vec<DiffLine>` into hunks with at most `context` surrounding
/// context lines around changed regions.
fn build_hunks_from_diff(diff: &[DiffLine], context: usize) -> Vec<DiffHunk> {
    if diff.is_empty() {
        return Vec::new();
    }

    // Find ranges of changed lines.
    let mut change_indices: Vec<usize> = diff
        .iter()
        .enumerate()
        .filter(|(_, l)| !matches!(l, DiffLine::Context(_)))
        .map(|(i, _)| i)
        .collect();

    if change_indices.is_empty() {
        return Vec::new();
    }

    // Group into contiguous regions (with context buffer).
    let mut hunks: Vec<DiffHunk> = Vec::new();
    let mut start = change_indices[0].saturating_sub(context);
    let mut end = (change_indices[0] + context + 1).min(diff.len());

    for &idx in &change_indices[1..] {
        let new_start = idx.saturating_sub(context);
        let new_end = (idx + context + 1).min(diff.len());
        if new_start <= end {
            // Merge into current hunk.
            end = new_end;
        } else {
            // Emit previous hunk.
            let old_count = diff[start..end]
                .iter()
                .filter(|l| !matches!(l, DiffLine::Added(_)))
                .count();
            let new_count = diff[start..end]
                .iter()
                .filter(|l| !matches!(l, DiffLine::Removed(_)))
                .count();
            let header = format!(
                "@@ -{},{} +{},{} @@",
                start + 1,
                old_count,
                start + 1,
                new_count
            );
            hunks.push(DiffHunk {
                header,
                lines: diff[start..end].to_vec(),
            });
            start = new_start;
            end = new_end;
        }
    }

    // Emit last hunk.
    let old_count = diff[start..end]
        .iter()
        .filter(|l| !matches!(l, DiffLine::Added(_)))
        .count();
    let new_count = diff[start..end]
        .iter()
        .filter(|l| !matches!(l, DiffLine::Removed(_)))
        .count();
    let header = format!(
        "@@ -{},{} +{},{} @@",
        start + 1,
        old_count,
        start + 1,
        new_count
    );
    hunks.push(DiffHunk {
        header,
        lines: diff[start..end].to_vec(),
    });

    hunks
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RepairAction, RepairPlan};

    fn sample_plan() -> RepairPlan {
        RepairPlan {
            actions: vec![
                RepairAction::reduce_retries("gateway", "orders", 3, 1),
                RepairAction::adjust_timeout("orders", "inventory", 5000, 3000),
            ],
            total_deviation: 3.0,
            affected_services: vec!["gateway".into(), "orders".into()],
            feasible: true,
        }
    }

    #[test]
    fn test_compute_line_diff_identical() {
        let old = vec!["a", "b", "c"];
        let diff = compute_line_diff(&old, &old);
        assert!(diff.iter().all(|l| matches!(l, DiffLine::Context(_))));
    }

    #[test]
    fn test_compute_line_diff_addition() {
        let old = vec!["a", "c"];
        let new = vec!["a", "b", "c"];
        let diff = compute_line_diff(&old, &new);
        assert!(diff.iter().any(|l| matches!(l, DiffLine::Added(s) if s == "b")));
    }

    #[test]
    fn test_compute_line_diff_removal() {
        let old = vec!["a", "b", "c"];
        let new = vec!["a", "c"];
        let diff = compute_line_diff(&old, &new);
        assert!(diff.iter().any(|l| matches!(l, DiffLine::Removed(s) if s == "b")));
    }

    #[test]
    fn test_compute_line_diff_replacement() {
        let old = vec!["a", "b", "c"];
        let new = vec!["a", "x", "c"];
        let diff = compute_line_diff(&old, &new);
        let has_removed = diff.iter().any(|l| matches!(l, DiffLine::Removed(s) if s == "b"));
        let has_added = diff.iter().any(|l| matches!(l, DiffLine::Added(s) if s == "x"));
        assert!(has_removed && has_added);
    }

    #[test]
    fn test_generate_diff_empty_plan() {
        let gen = ConfigDiffGenerator::new();
        let plan = RepairPlan::default();
        let diff = gen.generate_diff(&plan, &HashMap::new());
        assert!(diff.is_empty());
        assert_eq!(diff.summary, "No configuration changes required.");
    }

    #[test]
    fn test_generate_diff_with_actions() {
        let gen = ConfigDiffGenerator::new();
        let plan = sample_plan();
        let mut configs = HashMap::new();
        configs.insert(
            "gateway.yaml".to_string(),
            "# edge: gateway->orders\nattempts: 3\n".to_string(),
        );
        configs.insert(
            "orders.yaml".to_string(),
            "# edge: orders->inventory\nconnectTimeout: 5000ms\n".to_string(),
        );
        let diff = gen.generate_diff(&plan, &configs);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_generate_yaml_patch() {
        let gen = ConfigDiffGenerator::new();
        let plan = sample_plan();
        let patch = gen.generate_yaml_patch(&plan);
        assert!(patch.contains("attempts: 1"));
        assert!(patch.contains("connectTimeout: 3000ms"));
    }

    #[test]
    fn test_generate_unified_diff() {
        let gen = ConfigDiffGenerator::new();
        let original = "line1\nline2\nline3\n";
        let modified = "line1\nchanged\nline3\n";
        let diff = gen.generate_unified_diff(original, modified);
        assert!(diff.contains("--- a/original"));
        assert!(diff.contains("+++ b/modified"));
    }

    #[test]
    fn test_apply_repair_to_yaml_retry() {
        let gen = ConfigDiffGenerator::new();
        let yaml = "# edge: A->B\nattempts: 3\ntimeout: 1000\n";
        let action = RepairAction::reduce_retries("A", "B", 3, 1);
        let result = gen.apply_repair_to_yaml(yaml, &action);
        assert!(result.contains("attempts: 1"));
    }

    #[test]
    fn test_apply_repair_to_yaml_timeout() {
        let gen = ConfigDiffGenerator::new();
        let yaml = "# edge: A->B\nconnectTimeout: 5000ms\n";
        let action = RepairAction::adjust_timeout("A", "B", 5000, 2000);
        let result = gen.apply_repair_to_yaml(yaml, &action);
        assert!(result.contains("connectTimeout: 2000ms"));
    }

    #[test]
    fn test_format_diff_colored() {
        let gen = ConfigDiffGenerator::new();
        let diff = ConfigDiff {
            changes: vec![FileChange {
                file_path: "test.yaml".to_string(),
                hunks: vec![DiffHunk {
                    header: "@@ -1,2 +1,2 @@".to_string(),
                    lines: vec![
                        DiffLine::Removed("old line".to_string()),
                        DiffLine::Added("new line".to_string()),
                    ],
                }],
            }],
            summary: "1 file changed".to_string(),
        };
        let colored = gen.format_diff_colored(&diff);
        assert!(colored.contains("[-] old line"));
        assert!(colored.contains("[+] new line"));
    }

    #[test]
    fn test_config_diff_counts() {
        let diff = ConfigDiff {
            changes: vec![FileChange {
                file_path: "a.yaml".to_string(),
                hunks: vec![DiffHunk {
                    header: "@@".to_string(),
                    lines: vec![
                        DiffLine::Removed("x".into()),
                        DiffLine::Added("y".into()),
                        DiffLine::Added("z".into()),
                        DiffLine::Context("c".into()),
                    ],
                }],
            }],
            summary: "".to_string(),
        };
        assert_eq!(diff.lines_added(), 2);
        assert_eq!(diff.lines_removed(), 1);
    }

    #[test]
    fn test_diff_line_display() {
        assert_eq!(DiffLine::Context("a".into()).to_string(), " a");
        assert_eq!(DiffLine::Added("b".into()).to_string(), "+b");
        assert_eq!(DiffLine::Removed("c".into()).to_string(), "-c");
    }

    #[test]
    fn test_empty_inputs_diff() {
        let diff = compute_line_diff(&[], &[]);
        assert!(diff.is_empty());
    }
}

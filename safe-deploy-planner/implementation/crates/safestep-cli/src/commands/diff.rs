//! Implementation of the `diff` subcommand.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cli::DiffArgs;
use crate::output::OutputManager;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A generic diffable document (plan or schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffableDocument {
    #[serde(default)]
    pub plan_id: Option<String>,
    #[serde(default)]
    pub services: Vec<String>,
    #[serde(default)]
    pub steps: Vec<DiffableStep>,
    #[serde(default)]
    pub start_state: Vec<u16>,
    #[serde(default)]
    pub target_state: Vec<u16>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffableStep {
    pub step: usize,
    pub service: String,
    #[serde(default)]
    pub from_version: String,
    #[serde(default)]
    pub to_version: String,
    #[serde(default)]
    pub risk_score: u32,
}

/// Type of diff entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffType {
    Added,
    Removed,
    Modified,
    Unchanged,
}

impl std::fmt::Display for DiffType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Added => write!(f, "+"),
            Self::Removed => write!(f, "-"),
            Self::Modified => write!(f, "~"),
            Self::Unchanged => write!(f, " "),
        }
    }
}

/// A single diff entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    pub diff_type: DiffType,
    pub section: String,
    pub description: String,
    pub safety_relevant: bool,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

/// Full diff result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    pub old_source: String,
    pub new_source: String,
    pub entries: Vec<DiffEntry>,
    pub added_count: usize,
    pub removed_count: usize,
    pub modified_count: usize,
    pub safety_changes: usize,
}

// ---------------------------------------------------------------------------
// DiffCommand
// ---------------------------------------------------------------------------

pub struct DiffCommand {
    args: DiffArgs,
}

impl DiffCommand {
    pub fn new(args: DiffArgs) -> Self {
        Self { args }
    }

    pub fn execute(&self, output: &mut OutputManager) -> Result<()> {
        info!(old = %self.args.old.display(), new = %self.args.new.display(), "computing diff");

        let old_doc = self.load_document(&self.args.old)?;
        let new_doc = self.load_document(&self.args.new)?;

        let entries = self.compute_diff(&old_doc, &new_doc);

        let result = DiffResult {
            old_source: self.args.old.display().to_string(),
            new_source: self.args.new.display().to_string(),
            added_count: entries.iter().filter(|e| e.diff_type == DiffType::Added).count(),
            removed_count: entries.iter().filter(|e| e.diff_type == DiffType::Removed).count(),
            modified_count: entries.iter().filter(|e| e.diff_type == DiffType::Modified).count(),
            safety_changes: entries.iter().filter(|e| e.safety_relevant).count(),
            entries,
        };

        self.render_output(output, &result)?;
        Ok(())
    }

    fn load_document(&self, path: &std::path::Path) -> Result<DiffableDocument> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("json");
        let doc: DiffableDocument = if ext == "yaml" || ext == "yml" {
            serde_yaml::from_str(&content)
                .with_context(|| format!("failed to parse {}", path.display()))?
        } else {
            serde_json::from_str(&content)
                .with_context(|| format!("failed to parse {}", path.display()))?
        };
        Ok(doc)
    }

    fn compute_diff(&self, old: &DiffableDocument, new: &DiffableDocument) -> Vec<DiffEntry> {
        let mut entries = Vec::new();

        // Diff start state.
        if old.start_state != new.start_state {
            entries.push(DiffEntry {
                diff_type: DiffType::Modified,
                section: "start_state".into(),
                description: "start state changed".into(),
                safety_relevant: true,
                old_value: Some(format!("{:?}", old.start_state)),
                new_value: Some(format!("{:?}", new.start_state)),
            });
        }

        // Diff target state.
        if old.target_state != new.target_state {
            entries.push(DiffEntry {
                diff_type: DiffType::Modified,
                section: "target_state".into(),
                description: "target state changed".into(),
                safety_relevant: true,
                old_value: Some(format!("{:?}", old.target_state)),
                new_value: Some(format!("{:?}", new.target_state)),
            });
        }

        // Diff services.
        let old_services: std::collections::HashSet<&str> = old.services.iter().map(|s| s.as_str()).collect();
        let new_services: std::collections::HashSet<&str> = new.services.iter().map(|s| s.as_str()).collect();

        for svc in old_services.difference(&new_services) {
            entries.push(DiffEntry {
                diff_type: DiffType::Removed,
                section: "services".into(),
                description: format!("service '{}' removed", svc),
                safety_relevant: true,
                old_value: Some(svc.to_string()),
                new_value: None,
            });
        }

        for svc in new_services.difference(&old_services) {
            entries.push(DiffEntry {
                diff_type: DiffType::Added,
                section: "services".into(),
                description: format!("service '{}' added", svc),
                safety_relevant: false,
                old_value: None,
                new_value: Some(svc.to_string()),
            });
        }

        // Diff steps.
        let old_steps: HashMap<&str, &DiffableStep> = old.steps.iter()
            .map(|s| (s.service.as_str(), s)).collect();
        let new_steps: HashMap<&str, &DiffableStep> = new.steps.iter()
            .map(|s| (s.service.as_str(), s)).collect();

        for (svc, old_step) in &old_steps {
            if let Some(new_step) = new_steps.get(svc) {
                if old_step.from_version != new_step.from_version
                    || old_step.to_version != new_step.to_version
                {
                    entries.push(DiffEntry {
                        diff_type: DiffType::Modified,
                        section: format!("step/{}", svc),
                        description: format!(
                            "step for '{}' changed: {}->{} to {}->{}",
                            svc, old_step.from_version, old_step.to_version,
                            new_step.from_version, new_step.to_version
                        ),
                        safety_relevant: true,
                        old_value: Some(format!("{}->{}", old_step.from_version, old_step.to_version)),
                        new_value: Some(format!("{}->{}", new_step.from_version, new_step.to_version)),
                    });
                }

                if old_step.risk_score != new_step.risk_score {
                    let risk_increased = new_step.risk_score > old_step.risk_score;
                    entries.push(DiffEntry {
                        diff_type: DiffType::Modified,
                        section: format!("step/{}/risk", svc),
                        description: format!(
                            "risk for '{}' {} from {} to {}",
                            svc,
                            if risk_increased { "increased" } else { "decreased" },
                            old_step.risk_score, new_step.risk_score
                        ),
                        safety_relevant: risk_increased,
                        old_value: Some(old_step.risk_score.to_string()),
                        new_value: Some(new_step.risk_score.to_string()),
                    });
                }

                if old_step.step != new_step.step {
                    entries.push(DiffEntry {
                        diff_type: DiffType::Modified,
                        section: format!("step/{}/order", svc),
                        description: format!(
                            "order for '{}' changed from step {} to step {}",
                            svc, old_step.step, new_step.step
                        ),
                        safety_relevant: true,
                        old_value: Some(old_step.step.to_string()),
                        new_value: Some(new_step.step.to_string()),
                    });
                }
            } else {
                entries.push(DiffEntry {
                    diff_type: DiffType::Removed,
                    section: format!("step/{}", svc),
                    description: format!("step for service '{}' removed", svc),
                    safety_relevant: true,
                    old_value: Some(format!("step {}", old_step.step)),
                    new_value: None,
                });
            }
        }

        for (svc, new_step) in &new_steps {
            if !old_steps.contains_key(svc) {
                entries.push(DiffEntry {
                    diff_type: DiffType::Added,
                    section: format!("step/{}", svc),
                    description: format!("step for service '{}' added", svc),
                    safety_relevant: false,
                    old_value: None,
                    new_value: Some(format!("step {}", new_step.step)),
                });
            }
        }

        // Diff extra fields.
        for (key, old_val) in &old.extra {
            if let Some(new_val) = new.extra.get(key) {
                if old_val != new_val {
                    entries.push(DiffEntry {
                        diff_type: DiffType::Modified,
                        section: format!("extra/{}", key),
                        description: format!("field '{}' changed", key),
                        safety_relevant: false,
                        old_value: Some(old_val.to_string()),
                        new_value: Some(new_val.to_string()),
                    });
                }
            } else {
                entries.push(DiffEntry {
                    diff_type: DiffType::Removed,
                    section: format!("extra/{}", key),
                    description: format!("field '{}' removed", key),
                    safety_relevant: false,
                    old_value: Some(old_val.to_string()),
                    new_value: None,
                });
            }
        }

        for (key, new_val) in &new.extra {
            if !old.extra.contains_key(key) {
                entries.push(DiffEntry {
                    diff_type: DiffType::Added,
                    section: format!("extra/{}", key),
                    description: format!("field '{}' added", key),
                    safety_relevant: false,
                    old_value: None,
                    new_value: Some(new_val.to_string()),
                });
            }
        }

        entries
    }

    fn render_output(&self, output: &mut OutputManager, result: &DiffResult) -> Result<()> {
        let colors = output.colors().clone();

        output.section("Plan Diff");
        output.writeln(&format!("Old: {}", result.old_source));
        output.writeln(&format!("New: {}", result.new_source));
        output.writeln(&format!(
            "Changes: {} added, {} removed, {} modified",
            colors.safe(&result.added_count.to_string()),
            colors.error(&result.removed_count.to_string()),
            colors.warning(&result.modified_count.to_string()),
        ));

        if self.args.highlight_safety && result.safety_changes > 0 {
            output.writeln(&format!(
                "Safety-relevant changes: {}",
                colors.pnr(&result.safety_changes.to_string())
            ));
        }

        if result.entries.is_empty() {
            output.blank_line();
            output.writeln(&colors.safe("No differences found."));
            return Ok(());
        }

        output.blank_line();
        for entry in &result.entries {
            let marker = match entry.diff_type {
                DiffType::Added => colors.safe(&format!("+ {}", entry.description)),
                DiffType::Removed => colors.error(&format!("- {}", entry.description)),
                DiffType::Modified => colors.warning(&format!("~ {}", entry.description)),
                DiffType::Unchanged => colors.dim(&format!("  {}", entry.description)),
            };

            let safety_tag = if entry.safety_relevant && self.args.highlight_safety {
                format!(" {}", colors.pnr("[SAFETY]"))
            } else {
                String::new()
            };

            output.writeln(&format!("[{}] {}{}", entry.section, marker, safety_tag));

            if let (Some(ref old), Some(ref new)) = (&entry.old_value, &entry.new_value) {
                if entry.diff_type == DiffType::Modified {
                    output.writeln(&format!("  {} {}", colors.error(&format!("- {}", old)), ""));
                    output.writeln(&format!("  {} {}", colors.safe(&format!("+ {}", new)), ""));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::OutputFormat;
    use crate::output::OutputManager;

    fn sample_doc(plan_id: &str) -> DiffableDocument {
        DiffableDocument {
            plan_id: Some(plan_id.into()),
            services: vec!["api".into(), "db".into()],
            steps: vec![
                DiffableStep { step: 1, service: "api".into(), from_version: "v1".into(), to_version: "v2".into(), risk_score: 10 },
                DiffableStep { step: 2, service: "db".into(), from_version: "v1".into(), to_version: "v2".into(), risk_score: 5 },
            ],
            start_state: vec![0, 0],
            target_state: vec![1, 1],
            extra: HashMap::new(),
        }
    }

    #[test]
    fn test_diff_identical() {
        let cmd = DiffCommand::new(DiffArgs {
            old: "/tmp/a".into(), new: "/tmp/b".into(),
            highlight_safety: true, context_lines: 3,
        });
        let doc = sample_doc("plan-1");
        let entries = cmd.compute_diff(&doc, &doc);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_diff_changed_start_state() {
        let cmd = DiffCommand::new(DiffArgs {
            old: "/tmp/a".into(), new: "/tmp/b".into(),
            highlight_safety: true, context_lines: 3,
        });
        let old = sample_doc("plan-1");
        let mut new = sample_doc("plan-2");
        new.start_state = vec![1, 0];
        let entries = cmd.compute_diff(&old, &new);
        assert!(entries.iter().any(|e| e.section == "start_state" && e.safety_relevant));
    }

    #[test]
    fn test_diff_added_service() {
        let cmd = DiffCommand::new(DiffArgs {
            old: "/tmp/a".into(), new: "/tmp/b".into(),
            highlight_safety: true, context_lines: 3,
        });
        let old = sample_doc("plan-1");
        let mut new = sample_doc("plan-2");
        new.services.push("cache".into());
        let entries = cmd.compute_diff(&old, &new);
        assert!(entries.iter().any(|e| e.diff_type == DiffType::Added && e.description.contains("cache")));
    }

    #[test]
    fn test_diff_removed_step() {
        let cmd = DiffCommand::new(DiffArgs {
            old: "/tmp/a".into(), new: "/tmp/b".into(),
            highlight_safety: true, context_lines: 3,
        });
        let old = sample_doc("plan-1");
        let mut new = sample_doc("plan-2");
        new.steps.retain(|s| s.service != "db");
        let entries = cmd.compute_diff(&old, &new);
        assert!(entries.iter().any(|e| e.diff_type == DiffType::Removed && e.description.contains("db")));
    }

    #[test]
    fn test_diff_modified_risk() {
        let cmd = DiffCommand::new(DiffArgs {
            old: "/tmp/a".into(), new: "/tmp/b".into(),
            highlight_safety: true, context_lines: 3,
        });
        let old = sample_doc("plan-1");
        let mut new = sample_doc("plan-2");
        new.steps[0].risk_score = 50;
        let entries = cmd.compute_diff(&old, &new);
        assert!(entries.iter().any(|e| e.description.contains("risk") && e.safety_relevant));
    }

    #[test]
    fn test_diff_type_display() {
        assert_eq!(DiffType::Added.to_string(), "+");
        assert_eq!(DiffType::Removed.to_string(), "-");
        assert_eq!(DiffType::Modified.to_string(), "~");
        assert_eq!(DiffType::Unchanged.to_string(), " ");
    }

    #[test]
    fn test_diff_result_serialization() {
        let result = DiffResult {
            old_source: "old.json".into(),
            new_source: "new.json".into(),
            entries: vec![],
            added_count: 0,
            removed_count: 0,
            modified_count: 0,
            safety_changes: 0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: DiffResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.old_source, "old.json");
    }

    #[test]
    fn test_render_no_changes() {
        let cmd = DiffCommand::new(DiffArgs {
            old: "/tmp/a".into(), new: "/tmp/b".into(),
            highlight_safety: true, context_lines: 3,
        });
        let result = DiffResult {
            old_source: "a.json".into(), new_source: "b.json".into(),
            entries: vec![], added_count: 0, removed_count: 0, modified_count: 0, safety_changes: 0,
        };
        let mut output = OutputManager::new(OutputFormat::Text, false);
        cmd.render_output(&mut output, &result).unwrap();
        assert!(output.get_buffer().contains("No differences"));
    }

    #[test]
    fn test_render_with_changes() {
        let cmd = DiffCommand::new(DiffArgs {
            old: "/tmp/a".into(), new: "/tmp/b".into(),
            highlight_safety: true, context_lines: 3,
        });
        let result = DiffResult {
            old_source: "a.json".into(), new_source: "b.json".into(),
            entries: vec![
                DiffEntry {
                    diff_type: DiffType::Added, section: "services".into(),
                    description: "service added".into(), safety_relevant: false,
                    old_value: None, new_value: Some("cache".into()),
                },
            ],
            added_count: 1, removed_count: 0, modified_count: 0, safety_changes: 0,
        };
        let mut output = OutputManager::new(OutputFormat::Text, false);
        cmd.render_output(&mut output, &result).unwrap();
        assert!(output.get_buffer().contains("service added"));
    }
}

//! CI/CD integration for cascade analysis results.
//!
//! Provides gating policies, annotation generation for GitHub Actions and
//! GitLab CI, health-check output for ArgoCD, and diff-mode analysis that
//! scopes verification to changed services.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

use cascade_graph::rtig::RtigGraph;
use cascade_types::report::{Evidence, Finding, Location, Severity};
use cascade_types::service::ServiceId;

// ---------------------------------------------------------------------------
// Gating policy
// ---------------------------------------------------------------------------

/// Controls when the pipeline should block a deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingPolicy {
    /// Block if any critical-severity finding is present.
    pub fail_on_critical: bool,
    /// Block if any error-severity finding is present.
    pub fail_on_high: bool,
    /// Maximum number of medium (warning) findings before blocking.
    pub max_allowed_medium: usize,
    /// Custom rules evaluated in order; first match wins.
    pub custom_rules: Vec<GatingRule>,
}

impl Default for GatingPolicy {
    fn default() -> Self {
        Self {
            fail_on_critical: true,
            fail_on_high: true,
            max_allowed_medium: 5,
            custom_rules: Vec::new(),
        }
    }
}

/// A pattern-based gating rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingRule {
    pub severity: Severity,
    /// Substring pattern matched against the finding message.
    pub pattern: String,
    pub action: GatingAction,
}

/// What to do when a gating rule matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GatingAction {
    /// Fail the pipeline (exit code != 0).
    Block,
    /// Emit a warning annotation but do not fail.
    Warn,
    /// Suppress the finding entirely.
    Ignore,
}

// ---------------------------------------------------------------------------
// CiCdIntegration
// ---------------------------------------------------------------------------

/// Entry point for CI/CD-specific output and gating logic.
pub struct CiCdIntegration;

impl CiCdIntegration {
    /// Determine the process exit code based on findings and a gating policy.
    ///
    /// Returns 0 for clean, 1 for warnings-exceeded, 2 for errors/critical.
    pub fn determine_exit_code(findings: &[Finding], policy: &GatingPolicy) -> i32 {
        let effective: Vec<&Finding> = findings
            .iter()
            .filter(|f| {
                // Apply custom rules first.
                for rule in &policy.custom_rules {
                    if f.severity == rule.severity && f.description.contains(&rule.pattern) {
                        return rule.action == GatingAction::Block;
                    }
                }
                true
            })
            .collect();

        let has_critical = effective.iter().any(|f| f.severity == Severity::Critical);
        let has_error = effective.iter().any(|f| f.severity == Severity::High);
        let warning_count = effective
            .iter()
            .filter(|f| f.severity == Severity::Medium)
            .count();

        if has_critical && policy.fail_on_critical {
            return 2;
        }
        if has_error && policy.fail_on_high {
            return 2;
        }
        if warning_count > policy.max_allowed_medium {
            return 1;
        }
        0
    }

    /// Evaluate gating rules for a single finding.
    pub fn evaluate_finding(finding: &Finding, policy: &GatingPolicy) -> GatingAction {
        for rule in &policy.custom_rules {
            if finding.severity == rule.severity && finding.description.contains(&rule.pattern) {
                return rule.action;
            }
        }
        match finding.severity {
            Severity::Critical if policy.fail_on_critical => GatingAction::Block,
            Severity::High if policy.fail_on_high => GatingAction::Block,
            Severity::Medium => {
                // Warn by default; the caller aggregates.
                GatingAction::Warn
            }
            _ => GatingAction::Warn,
        }
    }
}

// ---------------------------------------------------------------------------
// GitHub Actions output
// ---------------------------------------------------------------------------

/// Annotation for GitHub Actions workflow commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubAnnotation {
    pub level: String,
    pub file: String,
    pub line: usize,
    pub end_line: usize,
    pub message: String,
    pub title: String,
}

/// Generates GitHub Actions-specific output (annotations, step summaries).
pub struct GitHubActionsOutput;

impl GitHubActionsOutput {
    /// Convert findings into GitHub Actions annotations.
    pub fn generate_annotations(findings: &[Finding]) -> Vec<GitHubAnnotation> {
        findings
            .iter()
            .map(|f| {
                let (file, line) = match &f.location.file {
                    Some(file) => (file.clone(), f.location.line.unwrap_or(1)),
                    None => Self::infer_location(f),
                };
                GitHubAnnotation {
                    level: Self::severity_to_level(f.severity).to_string(),
                    file,
                    line,
                    end_line: line,
                    message: f.description.clone(),
                    title: format!("CascadeVerify: {}", f.id),
                }
            })
            .collect()
    }

    /// Render annotations as GitHub Actions workflow commands.
    pub fn format_workflow_commands(annotations: &[GitHubAnnotation]) -> String {
        let mut out = String::new();
        for ann in annotations {
            let _ = writeln!(
                out,
                "::{level} file={file},line={line},endLine={end_line},title={title}::{message}",
                level = ann.level,
                file = ann.file,
                line = ann.line,
                end_line = ann.end_line,
                title = ann.title,
                message = ann.message,
            );
        }
        out
    }

    /// Generate a Markdown step summary for `$GITHUB_STEP_SUMMARY`.
    pub fn write_step_summary(findings: &[Finding], exit_code: i32) -> String {
        let mut md = String::with_capacity(2048);

        let critical = findings.iter().filter(|f| f.severity == Severity::Critical).count();
        let errors = findings.iter().filter(|f| f.severity == Severity::High).count();
        let warnings = findings.iter().filter(|f| f.severity == Severity::Medium).count();

        let _ = writeln!(md, "### CascadeVerify Results\n");

        if findings.is_empty() {
            let _ = writeln!(md, "> All cascade checks passed.\n");
        } else {
            let _ = writeln!(
                md,
                "| Severity | Count |\n|----------|-------|\n| Critical | {} |\n| Error | {} |\n| Warning | {} |",
                critical, errors, warnings
            );
            let _ = writeln!(md);

            let _ = writeln!(md, "| # | Severity | Finding |");
            let _ = writeln!(md, "|---|----------|---------|");
            for (i, f) in findings.iter().enumerate() {
                let badge = match f.severity {
                    Severity::Critical => "CRITICAL",
                    Severity::High => "ERROR",
                    Severity::Medium => "WARNING",
                    Severity::Low => "LOW",
                    Severity::Info => "INFO",
                };
                let _ = writeln!(
                    md,
                    "| {} | {} | {} |",
                    i + 1,
                    badge,
                    f.description.replace('|', "\\|")
                );
            }
        }

        let _ = writeln!(md, "\nExit code: `{}`", exit_code);
        md
    }

    fn severity_to_level(severity: Severity) -> &'static str {
        match severity {
            Severity::Critical | Severity::High => "error",
            Severity::Medium | Severity::Low => "warning",
            Severity::Info => "notice",
        }
    }

    fn infer_location(finding: &Finding) -> (String, usize) {
        for ev in &finding.evidence {
            if let Some(src) = &ev.source {
                return (src.clone(), 1);
            }
        }
        ("cascade-verify.yaml".to_string(), 1)
    }
}

// ---------------------------------------------------------------------------
// GitLab CI output
// ---------------------------------------------------------------------------

/// Generates GitLab Code Quality-compatible JSON.
pub struct GitLabCiOutput;

impl GitLabCiOutput {
    pub fn generate_code_quality(findings: &[Finding]) -> serde_json::Value {
        let items: Vec<serde_json::Value> = findings
            .iter()
            .map(|f| {
                let (file, line) = match &f.location.file {
                    Some(file) => (file.clone(), f.location.line.unwrap_or(1)),
                    None => ("cascade-verify.yaml".to_string(), 1),
                };
                serde_json::json!({
                    "type": "issue",
                    "check_name": f.id,
                    "description": f.description,
                    "severity": Self::severity_to_gitlab(f.severity),
                    "fingerprint": format!("cascade-verify:{}", f.id),
                    "location": {
                        "path": file,
                        "lines": {
                            "begin": line,
                        }
                    },
                    "categories": ["Security", "Reliability"],
                })
            })
            .collect();
        serde_json::Value::Array(items)
    }

    fn severity_to_gitlab(severity: Severity) -> &'static str {
        match severity {
            Severity::Critical => "critical",
            Severity::High => "major",
            Severity::Medium | Severity::Low => "minor",
            Severity::Info => "info",
        }
    }
}

// ---------------------------------------------------------------------------
// ArgoCD output
// ---------------------------------------------------------------------------

/// Generates ArgoCD-compatible health check JSON.
pub struct ArgoCdOutput;

impl ArgoCdOutput {
    pub fn generate_health_check(findings: &[Finding]) -> serde_json::Value {
        let has_critical = findings.iter().any(|f| f.severity == Severity::Critical);
        let has_error = findings.iter().any(|f| f.severity == Severity::High);

        let status = if has_critical {
            "Degraded"
        } else if has_error {
            "Progressing"
        } else {
            "Healthy"
        };

        let message = if findings.is_empty() {
            "All cascade verification checks passed".to_string()
        } else {
            format!(
                "{} cascade issue(s) detected ({} critical, {} error)",
                findings.len(),
                findings.iter().filter(|f| f.severity == Severity::Critical).count(),
                findings.iter().filter(|f| f.severity == Severity::High).count(),
            )
        };

        serde_json::json!({
            "status": status,
            "message": message,
            "details": {
                "total_findings": findings.len(),
                "critical": findings.iter().filter(|f| f.severity == Severity::Critical).count(),
                "error": findings.iter().filter(|f| f.severity == Severity::High).count(),
                "warning": findings.iter().filter(|f| f.severity == Severity::Medium).count(),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Diff-mode analysis
// ---------------------------------------------------------------------------

/// Scopes verification to only the services affected by a set of changed
/// files, reducing CI runtime on incremental changes.
pub struct DiffModeAnalyzer;

impl DiffModeAnalyzer {
    /// Given a list of changed file paths, identify which services are
    /// potentially affected.  Uses simple heuristics: filenames, directory
    /// names, and partial matches against service IDs in the graph.
    pub fn analyze_changed_files(
        changed: &[String],
        graph: &RtigGraph,
    ) -> Vec<ServiceId> {
        let service_ids: Vec<String> = graph
            .service_ids()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let mut affected = HashSet::new();

        for path in changed {
            let lower = path.to_lowercase();
            for svc in &service_ids {
                let svc_lower = svc.to_lowercase();
                // Match if the changed path contains the service name.
                if lower.contains(&svc_lower) {
                    affected.insert(svc.clone());
                }
            }
            // Also try to match the filename stem as a service name.
            if let Some(stem) = std::path::Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
            {
                let stem_lower = stem.to_lowercase();
                for svc in &service_ids {
                    if svc.to_lowercase() == stem_lower {
                        affected.insert(svc.clone());
                    }
                }
            }
        }

        affected.into_iter().map(ServiceId::new).collect()
    }

    /// Given a set of directly changed services, compute the full "affected
    /// cone" – all services that transitively depend on a changed service.
    pub fn compute_affected_cone_from_diff(
        changed_services: &[ServiceId],
        graph: &RtigGraph,
    ) -> HashSet<ServiceId> {
        let mut cone = HashSet::new();
        for svc in changed_services {
            cone.insert(svc.clone());
            let reachable = graph.forward_reachable(svc.as_str());
            for r in reachable {
                cone.insert(ServiceId::new(r));
            }
            let reverse = graph.reverse_reachable(svc.as_str());
            for r in reverse {
                cone.insert(ServiceId::new(r));
            }
        }
        cone
    }

    /// Filter findings to only those that mention services in the affected cone.
    pub fn filter_findings_to_cone(
        findings: &[Finding],
        cone: &HashSet<ServiceId>,
    ) -> Vec<Finding> {
        let cone_strs: HashSet<&str> = cone.iter().map(|s| s.as_str()).collect();
        findings
            .iter()
            .filter(|f| {
                // Check if any service in the cone is mentioned in the finding.
                let msg = &f.description;
                for svc in &cone_strs {
                    if msg.contains(svc) {
                        return true;
                    }
                }
                for ev in &f.evidence {
                    for svc in &cone_strs {
                        if ev.description.contains(svc) {
                            return true;
                        }
                    }
                }
                false
            })
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_graph::rtig::{DependencyEdgeInfo, ServiceNode};

    fn sample_findings() -> Vec<Finding> {
        vec![
            Finding {
                id: "AMP-001".into(),
                severity: Severity::Critical,
                title: "Amplification on gateway -> api".into(),
                description: "amplification on gateway -> api".into(),
                evidence: vec![],
                location: Location {
                    file: Some("envoy/gateway.yaml".into()),
                    service: None,
                    edge: None,
                    line: Some(10),
                    column: None,
                },
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "TMO-002".into(),
                severity: Severity::High,
                title: "Timeout on api -> auth".into(),
                description: "timeout on api -> auth".into(),
                evidence: vec![],
                location: Location {
                    file: None,
                    service: None,
                    edge: None,
                    line: None,
                    column: None,
                },
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "FAN-003".into(),
                severity: Severity::Medium,
                title: "Fan-in on db".into(),
                description: "fan-in on db".into(),
                evidence: vec![],
                location: Location {
                    file: None,
                    service: None,
                    edge: None,
                    line: None,
                    column: None,
                },
                code_flow: None,
                remediation: None,
            },
        ]
    }

    fn default_policy() -> GatingPolicy {
        GatingPolicy::default()
    }

    fn sample_graph() -> RtigGraph {
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("gateway", 1000));
        g.add_service(ServiceNode::new("api", 1000));
        g.add_service(ServiceNode::new("auth", 500));
        g.add_service(ServiceNode::new("db", 200));
        g.add_edge(DependencyEdgeInfo::new("gateway", "api").with_retry_count(3));
        g.add_edge(DependencyEdgeInfo::new("api", "auth").with_retry_count(2));
        g.add_edge(DependencyEdgeInfo::new("auth", "db").with_retry_count(2));
        g
    }

    #[test]
    fn test_exit_code_critical_blocks() {
        let code = CiCdIntegration::determine_exit_code(&sample_findings(), &default_policy());
        assert_eq!(code, 2);
    }

    #[test]
    fn test_exit_code_clean() {
        let code = CiCdIntegration::determine_exit_code(&[], &default_policy());
        assert_eq!(code, 0);
    }

    #[test]
    fn test_exit_code_warnings_only() {
        let findings = vec![Finding {
            id: "W1".into(),
            severity: Severity::Medium,
            title: "warn".into(),
            description: "warn".into(),
            evidence: vec![],
            location: Location {
                file: None,
                service: None,
                edge: None,
                line: None,
                column: None,
            },
            code_flow: None,
            remediation: None,
        }];
        let code = CiCdIntegration::determine_exit_code(&findings, &default_policy());
        assert_eq!(code, 0);
    }

    #[test]
    fn test_exit_code_too_many_warnings() {
        let findings: Vec<Finding> = (0..10)
            .map(|i| Finding {
                id: format!("W{}", i),
                severity: Severity::Medium,
                title: format!("warn {}", i),
                description: format!("warn {}", i),
                evidence: vec![],
                location: Location {
                    file: None,
                    service: None,
                    edge: None,
                    line: None,
                    column: None,
                },
                code_flow: None,
                remediation: None,
            })
            .collect();
        let code = CiCdIntegration::determine_exit_code(&findings, &default_policy());
        assert_eq!(code, 1);
    }

    #[test]
    fn test_custom_rule_ignore() {
        let policy = GatingPolicy {
            fail_on_critical: true,
            fail_on_high: true,
            max_allowed_medium: 5,
            custom_rules: vec![GatingRule {
                severity: Severity::Critical,
                pattern: "amplification".into(),
                action: GatingAction::Ignore,
            }],
        };
        let code = CiCdIntegration::determine_exit_code(&sample_findings(), &policy);
        // Critical is ignored by rule, but Error still triggers.
        assert_eq!(code, 2);
    }

    #[test]
    fn test_evaluate_finding() {
        let action =
            CiCdIntegration::evaluate_finding(&sample_findings()[0], &default_policy());
        assert_eq!(action, GatingAction::Block);
    }

    #[test]
    fn test_github_annotations() {
        let anns = GitHubActionsOutput::generate_annotations(&sample_findings());
        assert_eq!(anns.len(), 3);
        assert_eq!(anns[0].level, "error");
        assert_eq!(anns[0].file, "envoy/gateway.yaml");
    }

    #[test]
    fn test_github_workflow_commands() {
        let anns = GitHubActionsOutput::generate_annotations(&sample_findings());
        let cmds = GitHubActionsOutput::format_workflow_commands(&anns);
        assert!(cmds.contains("::error "));
        assert!(cmds.contains("::warning "));
    }

    #[test]
    fn test_github_step_summary() {
        let summary = GitHubActionsOutput::write_step_summary(&sample_findings(), 2);
        assert!(summary.contains("CascadeVerify Results"));
        assert!(summary.contains("Critical"));
    }

    #[test]
    fn test_github_step_summary_clean() {
        let summary = GitHubActionsOutput::write_step_summary(&[], 0);
        assert!(summary.contains("passed"));
    }

    #[test]
    fn test_gitlab_code_quality() {
        let cq = GitLabCiOutput::generate_code_quality(&sample_findings());
        let arr = cq.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["severity"], "critical");
        assert_eq!(arr[1]["severity"], "major");
    }

    #[test]
    fn test_argocd_health_critical() {
        let hc = ArgoCdOutput::generate_health_check(&sample_findings());
        assert_eq!(hc["status"], "Degraded");
    }

    #[test]
    fn test_argocd_health_clean() {
        let hc = ArgoCdOutput::generate_health_check(&[]);
        assert_eq!(hc["status"], "Healthy");
    }

    #[test]
    fn test_diff_mode_analyze_changed() {
        let g = sample_graph();
        let changed = vec!["envoy/gateway.yaml".into(), "istio/auth-dr.yaml".into()];
        let affected = DiffModeAnalyzer::analyze_changed_files(&changed, &g);
        let names: Vec<String> = affected.iter().map(|s| s.as_str().to_string()).collect();
        assert!(names.contains(&"gateway".to_string()));
        assert!(names.contains(&"auth".to_string()));
    }

    #[test]
    fn test_affected_cone() {
        let g = sample_graph();
        let changed = vec![ServiceId::new("api")];
        let cone = DiffModeAnalyzer::compute_affected_cone_from_diff(&changed, &g);
        // api -> auth -> db (forward), gateway -> api (reverse)
        assert!(cone.contains(&ServiceId::new("api")));
        assert!(cone.contains(&ServiceId::new("auth")));
        assert!(cone.contains(&ServiceId::new("db")));
        assert!(cone.contains(&ServiceId::new("gateway")));
    }

    #[test]
    fn test_filter_findings_to_cone() {
        let cone: HashSet<ServiceId> =
            vec![ServiceId::new("gateway"), ServiceId::new("api")]
                .into_iter()
                .collect();
        let filtered = DiffModeAnalyzer::filter_findings_to_cone(&sample_findings(), &cone);
        // Findings mentioning gateway or api should be kept; db-only dropped.
        assert!(filtered.iter().any(|f| f.description.contains("gateway")));
    }

    #[test]
    fn test_gating_policy_default() {
        let p = GatingPolicy::default();
        assert!(p.fail_on_critical);
        assert!(p.fail_on_high);
        assert_eq!(p.max_allowed_medium, 5);
    }
}

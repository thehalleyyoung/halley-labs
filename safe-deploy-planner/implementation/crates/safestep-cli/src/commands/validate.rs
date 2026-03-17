//! Implementation of the `validate` subcommand.
//!
//! Discovers Kubernetes manifest files in a directory tree, parses them as
//! JSON or YAML, and runs structural validation checks that catch common
//! authoring mistakes before they reach a cluster.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn};

use crate::cli::ValidateArgs;
use crate::commands::{CommandExecutor, Finding, FindingSeverity, render_findings};
use crate::output::OutputManager;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A manifest file that has been read from disk.
#[derive(Debug, Clone)]
pub struct ManifestFile {
    pub path: PathBuf,
    pub content: String,
    pub format: ManifestFormat,
}

/// The on-disk format of a manifest file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestFormat {
    Json,
    Yaml,
}

/// Aggregate result of the validation run, suitable for machine-readable output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub files_scanned: usize,
    pub files_valid: usize,
    pub errors: usize,
    pub warnings: usize,
    pub passed: bool,
}

// ---------------------------------------------------------------------------
// ValidateCommand
// ---------------------------------------------------------------------------

/// The validate sub-command.
pub struct ValidateCommand {
    args: ValidateArgs,
}

impl ValidateCommand {
    pub fn new(args: ValidateArgs) -> Self {
        Self { args }
    }

    /// Main entry-point called by the CLI dispatcher.
    fn run_validation(&self, out: &mut OutputManager) -> Result<ValidationResult> {
        let manifest_dir = &self.args.manifest_dir;

        info!(dir = %manifest_dir.display(), "starting manifest validation");

        let files = discover_files(manifest_dir, self.args.pattern.as_deref())
            .with_context(|| {
                format!(
                    "failed to discover manifest files in {}",
                    manifest_dir.display()
                )
            })?;

        out.section("Manifest Validation");
        out.writeln(&format!(
            "Scanning directory: {}",
            manifest_dir.display()
        ));
        if let Some(ref pat) = self.args.pattern {
            out.writeln(&format!("Filter pattern:    {pat}"));
        }
        out.writeln(&format!("Strict mode:       {}", self.args.strict));
        out.writeln(&format!("Max errors:        {}", self.args.max_errors));
        out.blank_line();

        let mut all_findings: Vec<Finding> = Vec::new();
        let mut files_valid: usize = 0;
        let mut total_errors: usize = 0;

        for mf in &files {
            let parsed: Result<Value, String> = match mf.format {
                ManifestFormat::Json => serde_json::from_str(&mf.content)
                    .map_err(|e| format!("JSON parse error: {e}")),
                ManifestFormat::Yaml => serde_yaml::from_str(&mf.content)
                    .map_err(|e| format!("YAML parse error: {e}")),
            };

            let file_display = mf.path.display().to_string();

            match parsed {
                Ok(value) => {
                    let mut findings = validate_manifest(&value, &file_display);

                    if self.args.strict {
                        for f in &mut findings {
                            if f.severity == FindingSeverity::Warning {
                                f.severity = FindingSeverity::Error;
                            }
                        }
                    }

                    let has_error = findings
                        .iter()
                        .any(|f| f.severity == FindingSeverity::Error);

                    if !has_error {
                        files_valid += 1;
                    }

                    for f in &findings {
                        if f.severity == FindingSeverity::Error {
                            total_errors += 1;
                        }
                    }

                    all_findings.extend(findings);
                }
                Err(msg) => {
                    all_findings.push(Finding {
                        severity: FindingSeverity::Error,
                        message: msg,
                        location: Some(file_display.clone()),
                        suggestion: None,
                    });
                    total_errors += 1;
                }
            }

            if total_errors >= self.args.max_errors {
                warn!(
                    max = self.args.max_errors,
                    "max error count reached – stopping early"
                );
                all_findings.push(Finding {
                    severity: FindingSeverity::Warning,
                    message: format!(
                        "Validation stopped early: reached max error count ({})",
                        self.args.max_errors
                    ),
                    location: None,
                    suggestion: None,
                });
                break;
            }
        }

        let warning_count = all_findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Warning)
            .count();

        let result = ValidationResult {
            files_scanned: files.len(),
            files_valid,
            errors: total_errors,
            warnings: warning_count,
            passed: total_errors == 0,
        };

        // -- render output ---------------------------------------------------
        out.section("Findings");

        if all_findings.is_empty() {
            out.writeln("No issues found – all manifests are valid.");
        } else {
            render_findings(out, &all_findings);
        }

        out.blank_line();
        out.section("Summary");

        let rows: Vec<Vec<String>> = vec![
            vec!["Files scanned".into(), result.files_scanned.to_string()],
            vec!["Files valid".into(), result.files_valid.to_string()],
            vec!["Errors".into(), result.errors.to_string()],
            vec!["Warnings".into(), result.warnings.to_string()],
            vec![
                "Result".into(),
                if result.passed {
                    "PASSED".into()
                } else {
                    "FAILED".into()
                },
            ],
        ];

        out.render_table(&["Metric", "Value"], &rows);
        out.blank_line();

        let serialized =
            serde_json::to_value(&result).context("failed to serialize validation result")?;
        out.render_value(&serialized);

        info!(
            passed = result.passed,
            errors = result.errors,
            warnings = result.warnings,
            "validation complete"
        );

        Ok(result)
    }
}

impl CommandExecutor for ValidateCommand {
    fn execute(&self, out: &mut OutputManager) -> Result<()> {
        let result = self.run_validation(out)?;
        if !result.passed {
            anyhow::bail!(
                "validation failed with {} error(s) and {} warning(s)",
                result.errors,
                result.warnings
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Recursively discover manifest files under `dir`, optionally filtering by a
/// simple glob pattern (applied against the file *name* only).
pub fn discover_files(dir: &Path, pattern: Option<&str>) -> Result<Vec<ManifestFile>> {
    let mut results = Vec::new();
    collect_files(dir, pattern, &mut results)?;
    results.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(results)
}

fn collect_files(
    dir: &Path,
    pattern: Option<&str>,
    out: &mut Vec<ManifestFile>,
) -> Result<()> {
    let entries = fs::read_dir(dir)
        .with_context(|| format!("cannot read directory {}", dir.display()))?;

    for entry in entries {
        let entry = entry.with_context(|| format!("error reading entry in {}", dir.display()))?;
        let path = entry.path();

        if path.is_dir() {
            collect_files(&path, pattern, out)?;
            continue;
        }

        let format = match path.extension().and_then(|e| e.to_str()) {
            Some("json") => ManifestFormat::Json,
            Some("yaml" | "yml") => ManifestFormat::Yaml,
            _ => continue,
        };

        if let Some(pat) = pattern {
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if !glob_matches(pat, file_name) {
                continue;
            }
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("cannot read file {}", path.display()))?;

        out.push(ManifestFile {
            path,
            content,
            format,
        });
    }
    Ok(())
}

/// Run structural validation checks on a parsed manifest value.
///
/// Returns a list of findings; an empty list means the manifest is valid.
pub fn validate_manifest(value: &Value, file_path: &str) -> Vec<Finding> {
    let mut findings: Vec<Finding> = Vec::new();

    let obj = match value.as_object() {
        Some(o) => o,
        None => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "manifest root is not a JSON object".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
            return findings;
        }
    };

    // -- apiVersion ----------------------------------------------------------
    match obj.get("apiVersion") {
        Some(Value::String(s)) if !s.is_empty() => {}
        Some(Value::String(_)) => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "apiVersion is an empty string".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
        Some(_) => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "apiVersion must be a string".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
        None => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "missing required field: apiVersion".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
    }

    // -- kind ----------------------------------------------------------------
    let kind_value: Option<&str> = match obj.get("kind") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.as_str()),
        Some(Value::String(_)) => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "kind is an empty string".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
            None
        }
        Some(_) => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "kind must be a string".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
            None
        }
        None => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "missing required field: kind".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
            None
        }
    };

    // -- metadata.name -------------------------------------------------------
    let metadata = obj.get("metadata").and_then(|m| m.as_object());
    match metadata.and_then(|m| m.get("name")) {
        Some(Value::String(s)) if !s.is_empty() => {}
        Some(Value::String(_)) => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "metadata.name is an empty string".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
        Some(_) => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "metadata.name must be a string".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
        None => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: "missing required field: metadata.name".into(),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
    }

    // -- metadata.namespace (for namespaced kinds) ---------------------------
    if let Some(kind) = kind_value {
        if is_namespaced_kind(kind) {
            let has_ns = metadata
                .and_then(|m| m.get("namespace"))
                .and_then(|v| v.as_str())
                .map(|s| !s.is_empty())
                .unwrap_or(false);

            if !has_ns {
                findings.push(Finding {
                    severity: FindingSeverity::Warning,
                    message: format!(
                        "namespaced resource kind '{kind}' is missing metadata.namespace"
                    ),
                    location: Some(file_path.into()),
                    suggestion: None,
                });
            }
        }
    }

    // -- workload-specific checks --------------------------------------------
    if let Some(kind) = kind_value {
        let workload_kinds: HashSet<&str> =
            ["Deployment", "StatefulSet", "DaemonSet"].iter().copied().collect();

        if workload_kinds.contains(kind) {
            validate_workload(value, kind, file_path, &mut findings);
        }
    }

    findings
}

/// Additional checks for Deployment / StatefulSet / DaemonSet manifests.
fn validate_workload(
    value: &Value,
    kind: &str,
    file_path: &str,
    findings: &mut Vec<Finding>,
) {
    let spec = value.get("spec");

    // -- spec.replicas (Deployment only) -------------------------------------
    if kind == "Deployment" {
        if let Some(replicas) = spec.and_then(|s| s.get("replicas")) {
            match replicas.as_u64() {
                Some(n) if n == 0 => {
                    findings.push(Finding {
                        severity: FindingSeverity::Warning,
                        message: "spec.replicas is 0".into(),
                        location: Some(file_path.into()),
                    suggestion: None,
                    });
                }
                Some(_) => { /* ok */ }
                None => {
                    findings.push(Finding {
                        severity: FindingSeverity::Error,
                        message: "spec.replicas must be a positive integer".into(),
                        location: Some(file_path.into()),
                    suggestion: None,
                    });
                }
            }
        }
    }

    // -- spec.template.spec.containers ---------------------------------------
    let containers = spec
        .and_then(|s| s.get("template"))
        .and_then(|t| t.get("spec"))
        .and_then(|s| s.get("containers"))
        .and_then(|c| c.as_array());

    match containers {
        Some(arr) if arr.is_empty() => {
            findings.push(Finding {
                severity: FindingSeverity::Warning,
                message: format!(
                    "{kind} has an empty containers list in spec.template.spec.containers"
                ),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
        Some(arr) => {
            for (i, container) in arr.iter().enumerate() {
                validate_container(container, i, file_path, findings);
            }
        }
        None => {
            findings.push(Finding {
                severity: FindingSeverity::Warning,
                message: format!(
                    "{kind} is missing spec.template.spec.containers"
                ),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
    }
}

/// Validate a single container definition inside a pod spec.
fn validate_container(
    container: &Value,
    index: usize,
    file_path: &str,
    findings: &mut Vec<Finding>,
) {
    let prefix = format!("container[{index}]");

    // -- image ---------------------------------------------------------------
    match container.get("image").and_then(|v| v.as_str()) {
        Some(img) if img.is_empty() => {
            findings.push(Finding {
                severity: FindingSeverity::Error,
                message: format!("{prefix}: image is an empty string"),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
        Some(_) => {}
        None => {
            findings.push(Finding {
                severity: FindingSeverity::Warning,
                message: format!("{prefix}: missing image field"),
                location: Some(file_path.into()),
                    suggestion: None,
            });
        }
    }

    // -- resource requests / limits ------------------------------------------
    if let Some(resources) = container.get("resources").and_then(|r| r.as_object()) {
        for section_name in &["requests", "limits"] {
            if let Some(section) = resources.get(*section_name).and_then(|s| s.as_object()) {
                for (key, val) in section {
                    if let Some(s) = val.as_str() {
                        if !validate_resource_string(s) {
                            findings.push(Finding {
                                severity: FindingSeverity::Warning,
                                message: format!(
                                    "{prefix}: invalid resource {section_name}.{key} value: \"{s}\""
                                ),
                                location: Some(file_path.into()),
                    suggestion: None,
                            });
                        }
                    }
                }
            }
        }
    }
}

/// Check whether a resource quantity string looks valid.
///
/// Accepted forms:
/// - Plain integers: `"4"`, `"128"`
/// - Plain decimals: `"0.5"`, `"1.5"`
/// - CPU millis: `"100m"`, `"250m"`
/// - Memory with binary suffix: `"256Mi"`, `"1Gi"`, `"1024Ki"`, `"2Ti"`
/// - Memory with SI suffix: `"500M"`, `"1G"`, `"100k"`
pub fn validate_resource_string(value: &str) -> bool {
    if value.is_empty() {
        return false;
    }

    // Plain number (integer or decimal)
    if value.parse::<f64>().is_ok() {
        return true;
    }

    // Millicpu: digits + 'm'
    if let Some(prefix) = value.strip_suffix('m') {
        return !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit());
    }

    // Binary suffixes: Ki, Mi, Gi, Ti, Pi, Ei
    let binary_suffixes = ["Ki", "Mi", "Gi", "Ti", "Pi", "Ei"];
    for suffix in &binary_suffixes {
        if let Some(prefix) = value.strip_suffix(suffix) {
            return !prefix.is_empty() && parse_decimal(prefix);
        }
    }

    // SI suffixes: k, M, G, T, P, E  (single character, case-sensitive)
    let si_suffixes = ['k', 'M', 'G', 'T', 'P', 'E'];
    if let Some(last) = value.chars().last() {
        if si_suffixes.contains(&last) {
            let prefix = &value[..value.len() - 1];
            return !prefix.is_empty() && parse_decimal(prefix);
        }
    }

    false
}

/// Return `true` if `s` is a valid non-negative decimal (integer or float).
fn parse_decimal(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut seen_dot = false;
    for ch in s.chars() {
        if ch == '.' {
            if seen_dot {
                return false;
            }
            seen_dot = true;
        } else if !ch.is_ascii_digit() {
            return false;
        }
    }
    true
}

/// Determine whether a Kubernetes resource kind is namespaced.
///
/// Cluster-scoped kinds (Namespace, ClusterRole, ClusterRoleBinding, etc.)
/// return `false`. Everything else is treated as namespaced.
pub fn is_namespaced_kind(kind: &str) -> bool {
    const CLUSTER_SCOPED: &[&str] = &[
        "Namespace",
        "Node",
        "PersistentVolume",
        "ClusterRole",
        "ClusterRoleBinding",
        "CustomResourceDefinition",
        "APIService",
        "MutatingWebhookConfiguration",
        "ValidatingWebhookConfiguration",
        "PriorityClass",
        "StorageClass",
        "CSIDriver",
        "CSINode",
        "VolumeAttachment",
        "ComponentStatus",
        "TokenReview",
        "SelfSubjectAccessReview",
        "SelfSubjectRulesReview",
        "SubjectAccessReview",
        "CertificateSigningRequest",
    ];
    !CLUSTER_SCOPED.contains(&kind)
}

/// Simple glob matching against a file name.
///
/// Supports `*` (matches any sequence of characters) and `?` (matches exactly
/// one character). All other characters are compared literally
/// (case-sensitive).
pub fn glob_matches(pattern: &str, filename: &str) -> bool {
    glob_match_recursive(pattern.as_bytes(), filename.as_bytes())
}

fn glob_match_recursive(pat: &[u8], name: &[u8]) -> bool {
    match (pat.first(), name.first()) {
        // Both exhausted → match.
        (None, None) => true,
        // Pattern has a trailing `*` that can match empty remainder.
        (Some(b'*'), None) => glob_match_recursive(&pat[1..], name),
        // Pattern exhausted but name is not → no match.
        (None, Some(_)) => false,
        // Wildcard `*`: try consuming zero or more characters from name.
        (Some(b'*'), Some(_)) => {
            // Either `*` matches nothing (advance pattern) …
            glob_match_recursive(&pat[1..], name)
                // … or `*` consumes one character from name.
                || glob_match_recursive(pat, &name[1..])
        }
        // Single-char wildcard.
        (Some(b'?'), Some(_)) => glob_match_recursive(&pat[1..], &name[1..]),
        // Literal character comparison.
        (Some(&pc), Some(&nc)) => {
            if pc == nc {
                glob_match_recursive(&pat[1..], &name[1..])
            } else {
                false
            }
        }
        // Pattern has remaining non-wildcard characters but name is exhausted.
        (Some(_), None) => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

    // -- glob matching -------------------------------------------------------

    #[test]
    fn glob_exact_match() {
        assert!(glob_matches("deploy.yaml", "deploy.yaml"));
    }

    #[test]
    fn glob_star_suffix() {
        assert!(glob_matches("deploy*", "deploy.yaml"));
        assert!(glob_matches("deploy*", "deploy-prod.yaml"));
    }

    #[test]
    fn glob_star_prefix() {
        assert!(glob_matches("*.yaml", "deploy.yaml"));
        assert!(!glob_matches("*.yaml", "deploy.json"));
    }

    #[test]
    fn glob_star_middle() {
        assert!(glob_matches("dep*.yaml", "deploy.yaml"));
        assert!(glob_matches("dep*.yaml", "dep.yaml"));
        assert!(!glob_matches("dep*.yaml", "service.yaml"));
    }

    #[test]
    fn glob_question_mark() {
        assert!(glob_matches("deploy?.yaml", "deploy1.yaml"));
        assert!(!glob_matches("deploy?.yaml", "deploy12.yaml"));
    }

    #[test]
    fn glob_no_match() {
        assert!(!glob_matches("service*", "deploy.yaml"));
    }

    #[test]
    fn glob_star_matches_empty() {
        assert!(glob_matches("deploy*.yaml", "deploy.yaml"));
    }

    // -- resource string validation ------------------------------------------

    #[test]
    fn resource_plain_integer() {
        assert!(validate_resource_string("4"));
        assert!(validate_resource_string("128"));
        assert!(validate_resource_string("0"));
    }

    #[test]
    fn resource_plain_decimal() {
        assert!(validate_resource_string("0.5"));
        assert!(validate_resource_string("1.5"));
    }

    #[test]
    fn resource_millicpu() {
        assert!(validate_resource_string("100m"));
        assert!(validate_resource_string("250m"));
        assert!(validate_resource_string("1000m"));
    }

    #[test]
    fn resource_binary_suffixes() {
        assert!(validate_resource_string("256Mi"));
        assert!(validate_resource_string("1Gi"));
        assert!(validate_resource_string("1024Ki"));
        assert!(validate_resource_string("2Ti"));
    }

    #[test]
    fn resource_si_suffixes() {
        assert!(validate_resource_string("500M"));
        assert!(validate_resource_string("1G"));
        assert!(validate_resource_string("100k"));
    }

    #[test]
    fn resource_invalid() {
        assert!(!validate_resource_string(""));
        assert!(!validate_resource_string("abc"));
        assert!(!validate_resource_string("Mi"));
        assert!(!validate_resource_string("100x"));
        assert!(!validate_resource_string("hello100m"));
    }

    // -- namespaced kind detection -------------------------------------------

    #[test]
    fn namespaced_kinds() {
        assert!(is_namespaced_kind("Deployment"));
        assert!(is_namespaced_kind("Service"));
        assert!(is_namespaced_kind("Pod"));
        assert!(is_namespaced_kind("ConfigMap"));
        assert!(is_namespaced_kind("Secret"));
        assert!(is_namespaced_kind("StatefulSet"));
        assert!(is_namespaced_kind("DaemonSet"));
    }

    #[test]
    fn cluster_scoped_kinds() {
        assert!(!is_namespaced_kind("Namespace"));
        assert!(!is_namespaced_kind("ClusterRole"));
        assert!(!is_namespaced_kind("ClusterRoleBinding"));
        assert!(!is_namespaced_kind("PersistentVolume"));
        assert!(!is_namespaced_kind("Node"));
        assert!(!is_namespaced_kind("CustomResourceDefinition"));
    }

    // -- manifest validation -------------------------------------------------

    #[test]
    fn valid_minimal_manifest() {
        let manifest = json!({
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": { "name": "test-ns" }
        });
        let findings = validate_manifest(&manifest, "ns.yaml");
        let errors: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Error)
            .collect();
        assert!(errors.is_empty(), "expected no errors: {errors:?}");
    }

    #[test]
    fn missing_api_version() {
        let manifest = json!({
            "kind": "Pod",
            "metadata": { "name": "test" }
        });
        let findings = validate_manifest(&manifest, "pod.yaml");
        assert!(findings.iter().any(|f| f.message.contains("apiVersion")));
    }

    #[test]
    fn missing_kind() {
        let manifest = json!({
            "apiVersion": "v1",
            "metadata": { "name": "test" }
        });
        let findings = validate_manifest(&manifest, "pod.yaml");
        assert!(findings.iter().any(|f| f.message.contains("kind")));
    }

    #[test]
    fn missing_metadata_name() {
        let manifest = json!({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {}
        });
        let findings = validate_manifest(&manifest, "pod.yaml");
        assert!(findings.iter().any(|f| f.message.contains("metadata.name")));
    }

    #[test]
    fn missing_namespace_warning() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": { "name": "web" },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [{ "name": "app", "image": "nginx" }]
                    }
                }
            }
        });
        let findings = validate_manifest(&manifest, "deploy.yaml");
        assert!(
            findings.iter().any(|f| f.severity == FindingSeverity::Warning
                && f.message.contains("namespace")),
            "expected namespace warning: {findings:?}"
        );
    }

    #[test]
    fn no_namespace_warning_for_cluster_scoped() {
        let manifest = json!({
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": { "name": "test-ns" }
        });
        let findings = validate_manifest(&manifest, "ns.yaml");
        assert!(
            !findings.iter().any(|f| f.message.contains("namespace")),
            "unexpected namespace warning for cluster-scoped kind: {findings:?}"
        );
    }

    #[test]
    fn deployment_missing_containers_warning() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": { "name": "web", "namespace": "default" },
            "spec": { "replicas": 1 }
        });
        let findings = validate_manifest(&manifest, "deploy.yaml");
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("containers")),
            "expected containers warning: {findings:?}"
        );
    }

    #[test]
    fn deployment_invalid_replicas() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": { "name": "web", "namespace": "default" },
            "spec": {
                "replicas": "three",
                "template": {
                    "spec": {
                        "containers": [{ "name": "app", "image": "nginx" }]
                    }
                }
            }
        });
        let findings = validate_manifest(&manifest, "deploy.yaml");
        assert!(
            findings
                .iter()
                .any(|f| f.severity == FindingSeverity::Error
                    && f.message.contains("replicas")),
            "expected replicas error: {findings:?}"
        );
    }

    #[test]
    fn deployment_zero_replicas_warning() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": { "name": "web", "namespace": "default" },
            "spec": {
                "replicas": 0,
                "template": {
                    "spec": {
                        "containers": [{ "name": "app", "image": "nginx" }]
                    }
                }
            }
        });
        let findings = validate_manifest(&manifest, "deploy.yaml");
        assert!(
            findings
                .iter()
                .any(|f| f.severity == FindingSeverity::Warning
                    && f.message.contains("replicas")),
            "expected zero-replicas warning: {findings:?}"
        );
    }

    #[test]
    fn container_empty_image() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": { "name": "web", "namespace": "default" },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [{ "name": "app", "image": "" }]
                    }
                }
            }
        });
        let findings = validate_manifest(&manifest, "deploy.yaml");
        assert!(
            findings
                .iter()
                .any(|f| f.severity == FindingSeverity::Error
                    && f.message.contains("image")),
            "expected empty image error: {findings:?}"
        );
    }

    #[test]
    fn container_invalid_resource() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": { "name": "web", "namespace": "default" },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "app",
                            "image": "nginx",
                            "resources": {
                                "requests": { "cpu": "100m", "memory": "badvalue" },
                                "limits": { "cpu": "200m", "memory": "512Mi" }
                            }
                        }]
                    }
                }
            }
        });
        let findings = validate_manifest(&manifest, "deploy.yaml");
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("badvalue")),
            "expected invalid resource warning: {findings:?}"
        );
    }

    #[test]
    fn container_valid_resources() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": { "name": "web", "namespace": "default" },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "app",
                            "image": "nginx",
                            "resources": {
                                "requests": { "cpu": "100m", "memory": "256Mi" },
                                "limits": { "cpu": "500m", "memory": "1Gi" }
                            }
                        }]
                    }
                }
            }
        });
        let findings = validate_manifest(&manifest, "deploy.yaml");
        let resource_issues: Vec<_> = findings
            .iter()
            .filter(|f| f.message.contains("resource") || f.message.contains("invalid"))
            .collect();
        assert!(
            resource_issues.is_empty(),
            "unexpected resource issues: {resource_issues:?}"
        );
    }

    #[test]
    fn non_object_root() {
        let manifest = json!("just a string");
        let findings = validate_manifest(&manifest, "bad.yaml");
        assert!(
            findings
                .iter()
                .any(|f| f.severity == FindingSeverity::Error
                    && f.message.contains("not a JSON object")),
            "expected root-type error: {findings:?}"
        );
    }

    #[test]
    fn api_version_wrong_type() {
        let manifest = json!({
            "apiVersion": 42,
            "kind": "Pod",
            "metadata": { "name": "test" }
        });
        let findings = validate_manifest(&manifest, "pod.yaml");
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("apiVersion must be a string")),
            "expected type error: {findings:?}"
        );
    }

    // -- file discovery (requires temp filesystem) ---------------------------

    fn create_temp_tree() -> TempDir {
        let dir = TempDir::new().unwrap();
        let root = dir.path();

        fs::write(
            root.join("deploy.yaml"),
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\n",
        )
        .unwrap();
        fs::write(
            root.join("service.yml"),
            "apiVersion: v1\nkind: Service\nmetadata:\n  name: web-svc\n",
        )
        .unwrap();
        fs::write(
            root.join("config.json"),
            r#"{"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"cfg"}}"#,
        )
        .unwrap();
        fs::write(root.join("readme.md"), "# ignored").unwrap();

        let sub = root.join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(
            sub.join("nested.yaml"),
            "apiVersion: v1\nkind: Pod\nmetadata:\n  name: busybox\n",
        )
        .unwrap();

        dir
    }

    #[test]
    fn discover_all_manifests() {
        let dir = create_temp_tree();
        let files = discover_files(dir.path(), None).unwrap();
        assert_eq!(files.len(), 4, "expected 4 manifest files: {files:?}");
    }

    #[test]
    fn discover_with_pattern() {
        let dir = create_temp_tree();
        let files = discover_files(dir.path(), Some("*.yaml")).unwrap();
        assert_eq!(files.len(), 2, "expected 2 .yaml files: {files:?}");
        assert!(files.iter().all(|f| f.format == ManifestFormat::Yaml));
    }

    #[test]
    fn discover_with_star_pattern() {
        let dir = create_temp_tree();
        let files = discover_files(dir.path(), Some("deploy*")).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].path.ends_with("deploy.yaml"));
    }

    #[test]
    fn discover_json_only() {
        let dir = create_temp_tree();
        let files = discover_files(dir.path(), Some("*.json")).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].format, ManifestFormat::Json);
    }

    #[test]
    fn discover_empty_dir() {
        let dir = TempDir::new().unwrap();
        let files = discover_files(dir.path(), None).unwrap();
        assert!(files.is_empty());
    }

    // -- strict mode ---------------------------------------------------------

    #[test]
    fn strict_mode_promotes_warnings() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": { "name": "web" },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [{ "name": "app", "image": "nginx" }]
                    }
                }
            }
        });

        let mut findings = validate_manifest(&manifest, "deploy.yaml");

        // Before promotion there should be warnings (e.g. missing namespace).
        let warnings_before = findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Warning)
            .count();
        assert!(warnings_before > 0, "expected at least one warning");

        // Simulate strict mode: promote warnings → errors.
        for f in &mut findings {
            if f.severity == FindingSeverity::Warning {
                f.severity = FindingSeverity::Error;
            }
        }

        let warnings_after = findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Warning)
            .count();
        assert_eq!(warnings_after, 0, "all warnings should be promoted");
    }

    // -- max errors cutoff ---------------------------------------------------

    #[test]
    fn max_errors_cutoff() {
        // Generate many bad manifests and verify we stop collecting after the
        // configured limit.
        let bad = json!({"not": "a manifest"});
        let max_errors: usize = 3;
        let total_files = 10;

        let mut all_findings: Vec<Finding> = Vec::new();
        let mut error_count: usize = 0;

        for i in 0..total_files {
            if error_count >= max_errors {
                break;
            }
            let findings = validate_manifest(&bad, &format!("file{i}.yaml"));
            for f in &findings {
                if f.severity == FindingSeverity::Error {
                    error_count += 1;
                }
            }
            all_findings.extend(findings);
        }

        assert!(
            error_count <= max_errors + 5,
            "error count {error_count} should be near the max ({max_errors})"
        );
        assert!(
            error_count >= max_errors,
            "should have accumulated at least {max_errors} errors"
        );
    }

    // -- finding generation --------------------------------------------------

    #[test]
    fn findings_include_location() {
        let manifest = json!({});
        let findings = validate_manifest(&manifest, "some/path/deploy.yaml");
        for f in &findings {
            assert_eq!(
                f.location.as_deref(),
                Some("some/path/deploy.yaml"),
                "finding should carry the file path"
            );
        }
    }

    #[test]
    fn fully_valid_deployment() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "frontend",
                "namespace": "production"
            },
            "spec": {
                "replicas": 3,
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "app",
                            "image": "registry.example.com/app:v1.2.3",
                            "resources": {
                                "requests": { "cpu": "100m", "memory": "256Mi" },
                                "limits": { "cpu": "500m", "memory": "512Mi" }
                            }
                        }]
                    }
                }
            }
        });
        let findings = validate_manifest(&manifest, "deploy.yaml");
        let errors: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "fully valid deployment should have no errors: {errors:?}"
        );
        let warnings: Vec<_> = findings
            .iter()
            .filter(|f| f.severity == FindingSeverity::Warning)
            .collect();
        assert!(
            warnings.is_empty(),
            "fully valid deployment should have no warnings: {warnings:?}"
        );
    }

    #[test]
    fn validation_result_serialization() {
        let result = ValidationResult {
            files_scanned: 5,
            files_valid: 3,
            errors: 2,
            warnings: 1,
            passed: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ValidationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.files_scanned, 5);
        assert_eq!(deserialized.files_valid, 3);
        assert_eq!(deserialized.errors, 2);
        assert_eq!(deserialized.warnings, 1);
        assert!(!deserialized.passed);
    }

    #[test]
    fn statefulset_missing_containers() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": { "name": "db", "namespace": "default" },
            "spec": { "replicas": 1 }
        });
        let findings = validate_manifest(&manifest, "sts.yaml");
        assert!(
            findings.iter().any(|f| f.message.contains("containers")),
            "StatefulSet should warn about missing containers: {findings:?}"
        );
    }

    #[test]
    fn daemonset_empty_containers() {
        let manifest = json!({
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": { "name": "agent", "namespace": "kube-system" },
            "spec": {
                "template": {
                    "spec": {
                        "containers": []
                    }
                }
            }
        });
        let findings = validate_manifest(&manifest, "ds.yaml");
        assert!(
            findings
                .iter()
                .any(|f| f.message.contains("empty containers")),
            "DaemonSet should warn about empty containers: {findings:?}"
        );
    }
}

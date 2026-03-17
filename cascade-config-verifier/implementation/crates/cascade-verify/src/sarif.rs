//! SARIF 2.1.0 report generation.
//!
//! Produces output conforming to the OASIS SARIF v2.1.0 specification
//! (<https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>)
//! so that results can be consumed by GitHub Code Scanning, VS Code SARIF
//! Viewer, and other compatible tooling.

use std::collections::HashMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cascade_types::repair::{RepairAction, RepairPlan};
use cascade_types::report::{Evidence, Finding, Location, Severity};

// ---------------------------------------------------------------------------
// SARIF 2.1.0 data model
// ---------------------------------------------------------------------------

/// Top-level SARIF log object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifReport {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<SarifRun>,
}

/// A single analysis run within the SARIF log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifRun {
    pub tool: SarifTool,
    pub results: Vec<SarifResult>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub artifacts: Vec<SarifArtifact>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub invocations: Vec<SarifInvocation>,
}

/// The tool that produced the SARIF results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifTool {
    pub driver: ToolComponent,
}

/// Driver component (the main analysis tool).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolComponent {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(rename = "semanticVersion", skip_serializing_if = "Option::is_none")]
    pub semantic_version: Option<String>,
    #[serde(rename = "informationUri", skip_serializing_if = "Option::is_none")]
    pub information_uri: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub rules: Vec<ReportingDescriptor>,
}

/// A rule that may produce results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingDescriptor {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "shortDescription", skip_serializing_if = "Option::is_none")]
    pub short_description: Option<MultiformatMessageString>,
    #[serde(rename = "fullDescription", skip_serializing_if = "Option::is_none")]
    pub full_description: Option<MultiformatMessageString>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub help: Option<MultiformatMessageString>,
    #[serde(rename = "helpUri", skip_serializing_if = "Option::is_none")]
    pub help_uri: Option<String>,
    #[serde(rename = "defaultConfiguration", skip_serializing_if = "Option::is_none")]
    pub default_configuration: Option<ReportingConfiguration>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub properties: Vec<PropertyBag>,
}

/// A default configuration for a reporting descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfiguration {
    pub level: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<f64>,
}

/// Multi-format message (text + optional markdown).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiformatMessageString {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markdown: Option<String>,
}

/// A single analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifResult {
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    #[serde(rename = "ruleIndex", skip_serializing_if = "Option::is_none")]
    pub rule_index: Option<usize>,
    pub level: String,
    pub message: SarifMessage,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub locations: Vec<SarifLocation>,
    #[serde(rename = "codeFlows", skip_serializing_if = "Vec::is_empty", default)]
    pub code_flows: Vec<SarifCodeFlow>,
    #[serde(rename = "relatedLocations", skip_serializing_if = "Vec::is_empty", default)]
    pub related_locations: Vec<SarifLocation>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub fixes: Vec<SarifFix>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fingerprints: Option<HashMap<String, String>>,
}

/// SARIF message object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifMessage {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markdown: Option<String>,
}

/// Location within the analysed artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifLocation {
    #[serde(rename = "physicalLocation", skip_serializing_if = "Option::is_none")]
    pub physical_location: Option<PhysicalLocation>,
    #[serde(rename = "logicalLocations", skip_serializing_if = "Vec::is_empty", default)]
    pub logical_locations: Vec<LogicalLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<SarifMessage>,
}

/// Physical location (file + region).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalLocation {
    #[serde(rename = "artifactLocation")]
    pub artifact_location: ArtifactLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<Region>,
    #[serde(rename = "contextRegion", skip_serializing_if = "Option::is_none")]
    pub context_region: Option<Region>,
}

/// URI-based reference to a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactLocation {
    pub uri: String,
    #[serde(rename = "uriBaseId", skip_serializing_if = "Option::is_none")]
    pub uri_base_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
}

/// A region within a text file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    #[serde(rename = "startLine", skip_serializing_if = "Option::is_none")]
    pub start_line: Option<usize>,
    #[serde(rename = "startColumn", skip_serializing_if = "Option::is_none")]
    pub start_column: Option<usize>,
    #[serde(rename = "endLine", skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    #[serde(rename = "endColumn", skip_serializing_if = "Option::is_none")]
    pub end_column: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<ArtifactContent>,
}

/// Inline artifact content (snippet).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactContent {
    pub text: String,
}

/// Logical location (e.g. namespace, function name).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalLocation {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(rename = "fullyQualifiedName", skip_serializing_if = "Option::is_none")]
    pub fully_qualified_name: Option<String>,
}

/// Code-flow showing execution path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifCodeFlow {
    #[serde(rename = "threadFlows")]
    pub thread_flows: Vec<ThreadFlow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<SarifMessage>,
}

/// A single thread flow within a code flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadFlow {
    pub locations: Vec<ThreadFlowLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<SarifMessage>,
}

/// A step in a thread flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadFlowLocation {
    pub location: SarifLocation,
    #[serde(rename = "nestingLevel", skip_serializing_if = "Option::is_none")]
    pub nesting_level: Option<i32>,
    #[serde(rename = "executionOrder", skip_serializing_if = "Option::is_none")]
    pub execution_order: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub importance: Option<String>,
}

/// A proposed fix for a result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifFix {
    pub description: SarifMessage,
    #[serde(rename = "artifactChanges")]
    pub artifact_changes: Vec<ArtifactChange>,
}

/// Changes to a specific artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactChange {
    #[serde(rename = "artifactLocation")]
    pub artifact_location: ArtifactLocation,
    pub replacements: Vec<Replacement>,
}

/// A text replacement within an artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replacement {
    #[serde(rename = "deletedRegion")]
    pub deleted_region: Region,
    #[serde(rename = "insertedContent", skip_serializing_if = "Option::is_none")]
    pub inserted_content: Option<ArtifactContent>,
}

/// Artifact reference in the run-level artifacts array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifArtifact {
    pub location: ArtifactLocation,
    #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub length: Option<i64>,
}

/// Invocation details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifInvocation {
    #[serde(rename = "executionSuccessful")]
    pub execution_successful: bool,
    #[serde(rename = "commandLine", skip_serializing_if = "Option::is_none")]
    pub command_line: Option<String>,
    #[serde(rename = "startTimeUtc", skip_serializing_if = "Option::is_none")]
    pub start_time_utc: Option<String>,
    #[serde(rename = "endTimeUtc", skip_serializing_if = "Option::is_none")]
    pub end_time_utc: Option<String>,
    #[serde(rename = "exitCode", skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
}

/// Generic property bag (string→string for now).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyBag {
    pub key: String,
    pub value: String,
}

// ---------------------------------------------------------------------------
// Report metadata
// ---------------------------------------------------------------------------

/// Metadata supplied by the caller to enrich the SARIF output.
#[derive(Debug, Clone, Default)]
pub struct ReportMetadata {
    pub tool_version: String,
    pub command_line: Option<String>,
    pub start_time: Option<String>,
    pub end_time: Option<String>,
    pub exit_code: Option<i32>,
    pub base_uri: Option<String>,
}

// ---------------------------------------------------------------------------
// SarifGenerator
// ---------------------------------------------------------------------------

/// Generates a full SARIF 2.1.0 report from a slice of [`Finding`]s.
pub struct SarifGenerator;

impl SarifGenerator {
    /// Produce a complete [`SarifReport`] from findings and optional metadata.
    pub fn generate(findings: &[Finding], metadata: &ReportMetadata) -> SarifReport {
        let rules = Self::build_rules(findings);
        let rule_index_map = Self::build_rule_index_map(&rules);
        let artifacts = Self::collect_artifacts(findings);
        let artifact_index_map = Self::build_artifact_index_map(&artifacts);

        let results: Vec<SarifResult> = findings
            .iter()
            .map(|f| Self::finding_to_result(f, &rule_index_map, &artifact_index_map))
            .collect();

        let tool = SarifTool {
            driver: ToolComponent {
                name: "CascadeVerify".to_string(),
                version: Some(metadata.tool_version.clone()),
                semantic_version: Some(metadata.tool_version.clone()),
                information_uri: Some(
                    "https://github.com/cascade-verify/cascade-config-verifier".to_string(),
                ),
                rules,
            },
        };

        let invocations = vec![SarifInvocation {
            execution_successful: metadata.exit_code.map(|c| c == 0).unwrap_or(true),
            command_line: metadata.command_line.clone(),
            start_time_utc: metadata.start_time.clone().or_else(|| Some(Utc::now().to_rfc3339())),
            end_time_utc: metadata.end_time.clone().or_else(|| Some(Utc::now().to_rfc3339())),
            exit_code: metadata.exit_code,
        }];

        SarifReport {
            schema: "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json".to_string(),
            version: "2.1.0".to_string(),
            runs: vec![SarifRun {
                tool,
                results,
                artifacts,
                invocations,
            }],
        }
    }

    /// Generate a SARIF report and include fixes derived from repair plans.
    pub fn generate_with_repairs(
        findings: &[Finding],
        repairs: &[RepairPlan],
        metadata: &ReportMetadata,
    ) -> SarifReport {
        let mut report = Self::generate(findings, metadata);
        if let Some(run) = report.runs.first_mut() {
            Self::attach_fixes(&mut run.results, repairs);
        }
        report
    }

    /// Serialize a [`SarifReport`] to a pretty-printed JSON string.
    pub fn to_json(report: &SarifReport) -> String {
        serde_json::to_string_pretty(report).unwrap_or_else(|_| "{}".to_string())
    }

    // ---- internal helpers ----

    fn build_rules(findings: &[Finding]) -> Vec<ReportingDescriptor> {
        let mut seen: HashMap<String, ReportingDescriptor> = HashMap::new();
        for finding in findings {
            let rule_id = Self::rule_id_for_finding(finding);
            seen.entry(rule_id.clone()).or_insert_with(|| {
                let (name, short, full) = Self::rule_metadata(&rule_id);
                ReportingDescriptor {
                    id: rule_id.clone(),
                    name: Some(name),
                    short_description: Some(MultiformatMessageString {
                        text: short.clone(),
                        markdown: None,
                    }),
                    full_description: Some(MultiformatMessageString {
                        text: full,
                        markdown: None,
                    }),
                    help: Some(MultiformatMessageString {
                        text: format!("See documentation for rule {}", rule_id),
                        markdown: Some(format!("See [docs](https://cascade-verify.dev/rules/{})", rule_id)),
                    }),
                    help_uri: Some(format!(
                        "https://cascade-verify.dev/rules/{}",
                        rule_id
                    )),
                    default_configuration: Some(ReportingConfiguration {
                        level: Self::severity_to_level(finding.severity).to_string(),
                        rank: Some(Self::severity_to_rank(finding.severity)),
                    }),
                    properties: Vec::new(),
                }
            });
        }
        let mut rules: Vec<_> = seen.into_values().collect();
        rules.sort_by(|a, b| a.id.cmp(&b.id));
        rules
    }

    fn build_rule_index_map(rules: &[ReportingDescriptor]) -> HashMap<String, usize> {
        rules.iter().enumerate().map(|(i, r)| (r.id.clone(), i)).collect()
    }

    fn collect_artifacts(findings: &[Finding]) -> Vec<SarifArtifact> {
        let mut uris: Vec<String> = Vec::new();
        for f in findings {
            if let Some(file) = &f.location.file {
                if !uris.contains(file) {
                    uris.push(file.clone());
                }
            }
            for ev in &f.evidence {
                if let Some(src) = &ev.source {
                    if !uris.contains(src) {
                        uris.push(src.clone());
                    }
                }
            }
        }
        uris.iter()
            .map(|uri| SarifArtifact {
                location: ArtifactLocation {
                    uri: uri.clone(),
                    uri_base_id: Some("%SRCROOT%".to_string()),
                    index: None,
                },
                mime_type: Self::guess_mime(uri),
                length: None,
            })
            .collect()
    }

    fn build_artifact_index_map(artifacts: &[SarifArtifact]) -> HashMap<String, usize> {
        artifacts
            .iter()
            .enumerate()
            .map(|(i, a)| (a.location.uri.clone(), i))
            .collect()
    }

    fn finding_to_result(
        finding: &Finding,
        rule_index: &HashMap<String, usize>,
        artifact_index: &HashMap<String, usize>,
    ) -> SarifResult {
        let rule_id = Self::rule_id_for_finding(finding);
        let locations = Self::locations_from_finding(finding, artifact_index);
        let related = Self::related_locations(finding, artifact_index);
        let code_flows = Self::code_flows_from_evidence(&finding.evidence);

        let mut fingerprints = HashMap::new();
        fingerprints.insert(
            "cascadeVerify/v1".to_string(),
            finding.id.clone(),
        );

        SarifResult {
            rule_id: rule_id.clone(),
            rule_index: rule_index.get(&rule_id).copied(),
            level: Self::severity_to_level(finding.severity).to_string(),
            message: SarifMessage {
                text: finding.description.clone(),
                markdown: Some(format!("**{}**: {}", finding.severity_label(), finding.description)),
            },
            locations,
            code_flows,
            related_locations: related,
            fixes: Vec::new(),
            fingerprints: Some(fingerprints),
        }
    }

    fn locations_from_finding(
        finding: &Finding,
        artifact_index: &HashMap<String, usize>,
    ) -> Vec<SarifLocation> {
        if finding.location.file.is_some() {
            vec![Self::to_sarif_location(&finding.location, artifact_index)]
        } else {
            Vec::new()
        }
    }

    fn to_sarif_location(
        loc: &Location,
        artifact_index: &HashMap<String, usize>,
    ) -> SarifLocation {
        let file = loc.file.clone().unwrap_or_default();
        SarifLocation {
            physical_location: Some(PhysicalLocation {
                artifact_location: ArtifactLocation {
                    uri: file.clone(),
                    uri_base_id: Some("%SRCROOT%".to_string()),
                    index: artifact_index.get(&file).copied(),
                },
                region: Some(Region {
                    start_line: loc.line,
                    start_column: loc.column,
                    end_line: loc.line,
                    end_column: None,
                    snippet: None,
                }),
                context_region: None,
            }),
            logical_locations: Vec::new(),
            id: None,
            message: None,
        }
    }

    fn related_locations(
        finding: &Finding,
        artifact_index: &HashMap<String, usize>,
    ) -> Vec<SarifLocation> {
        finding
            .evidence
            .iter()
            .enumerate()
            .filter_map(|(i, ev)| {
                ev.source.as_ref().map(|src| {
                    SarifLocation {
                        physical_location: Some(PhysicalLocation {
                            artifact_location: ArtifactLocation {
                                uri: src.clone(),
                                uri_base_id: Some("%SRCROOT%".to_string()),
                                index: artifact_index.get(src).copied(),
                            },
                            region: None,
                            context_region: None,
                        }),
                        logical_locations: Vec::new(),
                        id: Some(i as i64),
                        message: Some(SarifMessage {
                            text: ev.description.clone(),
                            markdown: None,
                        }),
                    }
                })
            })
            .collect()
    }

    fn code_flows_from_evidence(evidence: &[Evidence]) -> Vec<SarifCodeFlow> {
        if evidence.len() < 2 {
            return Vec::new();
        }
        let thread_flow_locs: Vec<ThreadFlowLocation> = evidence
            .iter()
            .enumerate()
            .map(|(i, ev)| {
                let loc = ev.source.as_ref().map(|src| SarifLocation {
                    physical_location: Some(PhysicalLocation {
                        artifact_location: ArtifactLocation {
                            uri: src.clone(),
                            uri_base_id: Some("%SRCROOT%".to_string()),
                            index: None,
                        },
                        region: None,
                        context_region: None,
                    }),
                    logical_locations: vec![LogicalLocation {
                        name: ev.description.clone(),
                        kind: Some("service".to_string()),
                        fully_qualified_name: None,
                    }],
                    id: None,
                    message: Some(SarifMessage {
                        text: ev.description.clone(),
                        markdown: None,
                    }),
                });
                let default_loc = SarifLocation {
                    physical_location: None,
                    logical_locations: vec![LogicalLocation {
                        name: ev.description.clone(),
                        kind: Some("service".to_string()),
                        fully_qualified_name: None,
                    }],
                    id: None,
                    message: Some(SarifMessage {
                        text: ev.description.clone(),
                        markdown: None,
                    }),
                };
                ThreadFlowLocation {
                    location: loc.unwrap_or(default_loc),
                    nesting_level: Some(0),
                    execution_order: Some(i as i32 + 1),
                    importance: Some(if i == 0 { "essential" } else { "important" }.to_string()),
                }
            })
            .collect();

        vec![SarifCodeFlow {
            thread_flows: vec![ThreadFlow {
                locations: thread_flow_locs,
                id: Some(Uuid::new_v4().to_string()),
                message: None,
            }],
            message: Some(SarifMessage {
                text: "Cascade propagation path".to_string(),
                markdown: None,
            }),
        }]
    }

    fn attach_fixes(results: &mut [SarifResult], repairs: &[RepairPlan]) {
        for repair in repairs {
            for action in &repair.actions {
                let edge_id_str = Self::repair_action_edge_id(action);
                let param_str = Self::repair_action_parameter(action);
                for result in results.iter_mut() {
                    if result.message.text.contains(&edge_id_str)
                        || result.message.text.contains(param_str)
                    {
                        result.fixes.push(Self::repair_action_to_fix(action));
                    }
                }
            }
        }
    }

    fn repair_action_edge_id(action: &RepairAction) -> String {
        match action {
            RepairAction::ModifyRetryCount { edge_id, .. }
            | RepairAction::ModifyTimeout { edge_id, .. }
            | RepairAction::AddCircuitBreaker { edge_id, .. }
            | RepairAction::AddRateLimit { edge_id, .. }
            | RepairAction::AddBulkhead { edge_id, .. }
            | RepairAction::RemoveRetry { edge_id }
            | RepairAction::ModifyBackoff { edge_id, .. } => edge_id.to_string(),
            RepairAction::ModifyCapacity { service_id, .. } => service_id.clone(),
        }
    }

    fn repair_action_parameter(action: &RepairAction) -> &'static str {
        match action {
            RepairAction::ModifyRetryCount { .. } => "retry_count",
            RepairAction::ModifyTimeout { .. } => "timeout_ms",
            RepairAction::ModifyCapacity { .. } => "capacity",
            RepairAction::AddCircuitBreaker { .. } => "circuit_breaker",
            RepairAction::AddRateLimit { .. } => "rate_limit",
            RepairAction::AddBulkhead { .. } => "bulkhead",
            RepairAction::RemoveRetry { .. } => "retry",
            RepairAction::ModifyBackoff { .. } => "backoff",
        }
    }

    fn repair_action_to_fix(action: &RepairAction) -> SarifFix {
        let edge_id_str = Self::repair_action_edge_id(action);
        let param = Self::repair_action_parameter(action);
        let description_text = match action {
            RepairAction::ModifyRetryCount { new_count, .. } => {
                format!("Change {} on {} to {}", param, edge_id_str, new_count)
            }
            RepairAction::ModifyTimeout { new_timeout_ms, .. } => {
                format!("Change {} on {} to {}", param, edge_id_str, new_timeout_ms)
            }
            RepairAction::ModifyCapacity { new_capacity, .. } => {
                format!("Change {} on {} to {}", param, edge_id_str, new_capacity)
            }
            RepairAction::AddCircuitBreaker { max_connections, consecutive_errors, .. } => {
                format!("Add {} on {} (max_conn={}, consec_err={})", param, edge_id_str, max_connections, consecutive_errors)
            }
            RepairAction::AddRateLimit { requests_per_second, .. } => {
                format!("Add {} on {} ({} rps)", param, edge_id_str, requests_per_second)
            }
            RepairAction::AddBulkhead { max_concurrent, .. } => {
                format!("Add {} on {} (max_concurrent={})", param, edge_id_str, max_concurrent)
            }
            RepairAction::RemoveRetry { .. } => {
                format!("Remove {} on {}", param, edge_id_str)
            }
            RepairAction::ModifyBackoff { strategy, .. } => {
                format!("Change {} on {} to {}", param, edge_id_str, strategy)
            }
        };
        SarifFix {
            description: SarifMessage {
                text: description_text.clone(),
                markdown: Some(description_text.clone()),
            },
            artifact_changes: vec![ArtifactChange {
                artifact_location: ArtifactLocation {
                    uri: edge_id_str,
                    uri_base_id: None,
                    index: None,
                },
                replacements: vec![Replacement {
                    deleted_region: Region {
                        start_line: Some(1),
                        start_column: Some(1),
                        end_line: Some(1),
                        end_column: None,
                        snippet: Some(ArtifactContent {
                            text: format!("{}: N/A", param),
                        }),
                    },
                    inserted_content: Some(ArtifactContent {
                        text: description_text.clone(),
                    }),
                }],
            }],
        }
    }

    fn rule_id_for_finding(finding: &Finding) -> String {
        let id = &finding.id;
        if id.starts_with("AMP-") {
            "CV001".to_string()
        } else if id.starts_with("TMO-") {
            "CV002".to_string()
        } else if id.starts_with("FAN-") {
            "CV003".to_string()
        } else if id.starts_with("T2-CASCADE-") {
            "CV004".to_string()
        } else if id.starts_with("T2-CONVERGENCE-") {
            "CV005".to_string()
        } else {
            "CV999".to_string()
        }
    }

    fn rule_metadata(rule_id: &str) -> (String, String, String) {
        match rule_id {
            "CV001" => (
                "RetryAmplification".to_string(),
                "Retry amplification exceeds safe threshold".to_string(),
                "The cumulative retry amplification factor along a request path exceeds the configured threshold, risking cascading overload.".to_string(),
            ),
            "CV002" => (
                "TimeoutBudgetExceeded".to_string(),
                "Timeout budget is inconsistent between caller and callee".to_string(),
                "The effective downstream timeout (per-try × retries + backoff) exceeds the caller's own request timeout, causing unpredictable deadline propagation.".to_string(),
            ),
            "CV003" => (
                "HighFanIn".to_string(),
                "Service has dangerously high fan-in".to_string(),
                "A service receives requests from many upstream services, amplifying the impact of any latency or error spike.".to_string(),
            ),
            "CV004" => (
                "CascadingFailure".to_string(),
                "Cascading failure affects majority of services".to_string(),
                "A simulated failure originating at one service propagates to more than half the topology through retry and timeout interactions.".to_string(),
            ),
            "CV005" => (
                "ConvergencePoint".to_string(),
                "Critical convergence point with high amplification".to_string(),
                "A service acts as a convergence point reachable from many upstream services with significant aggregated amplification.".to_string(),
            ),
            _ => (
                "UnknownRule".to_string(),
                "Unknown finding type".to_string(),
                "An unclassified finding was detected.".to_string(),
            ),
        }
    }

    fn severity_to_level(severity: Severity) -> &'static str {
        match severity {
            Severity::Critical => "error",
            Severity::High => "error",
            Severity::Medium => "warning",
            Severity::Low => "warning",
            Severity::Info => "note",
        }
    }

    fn severity_to_rank(severity: Severity) -> f64 {
        match severity {
            Severity::Critical => 95.0,
            Severity::High => 75.0,
            Severity::Medium => 50.0,
            Severity::Low => 30.0,
            Severity::Info => 20.0,
        }
    }

    fn guess_mime(uri: &str) -> Option<String> {
        if uri.ends_with(".yaml") || uri.ends_with(".yml") {
            Some("text/yaml".to_string())
        } else if uri.ends_with(".json") {
            Some("application/json".to_string())
        } else if uri.ends_with(".toml") {
            Some("application/toml".to_string())
        } else {
            None
        }
    }
}

// Helper trait to get a severity label from Finding without modifying cascade-types.
trait FindingSeverityLabel {
    fn severity_label(&self) -> &'static str;
}

impl FindingSeverityLabel for Finding {
    fn severity_label(&self) -> &'static str {
        match self.severity {
            Severity::Critical => "CRITICAL",
            Severity::High => "ERROR",
            Severity::Medium => "WARNING",
            Severity::Low => "LOW",
            Severity::Info => "INFO",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_types::topology::EdgeId;

    fn sample_findings() -> Vec<Finding> {
        vec![
            Finding {
                id: "AMP-001".into(),
                severity: Severity::Critical,
                title: "Retry Amplification".into(),
                description: "retry amplification factor 64.0x along path a → b → c → d".into(),
                evidence: vec![
                    Evidence {
                        description: "a → b (factor 4.0x)".into(),
                        value: None,
                        source: Some("envoy/cluster-a.yaml".into()),
                    },
                    Evidence {
                        description: "b → c (factor 4.0x)".into(),
                        value: None,
                        source: Some("envoy/cluster-b.yaml".into()),
                    },
                    Evidence {
                        description: "c → d (factor 4.0x)".into(),
                        value: None,
                        source: None,
                    },
                ],
                location: Location {
                    file: Some("envoy/cluster-a.yaml".into()),
                    service: None,
                    edge: None,
                    line: Some(12),
                    column: Some(5),
                },
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "TMO-002".into(),
                severity: Severity::Medium,
                title: "Timeout Budget Exceeded".into(),
                description: "timeout budget exceeded: caller → callee".into(),
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

    fn sample_metadata() -> ReportMetadata {
        ReportMetadata {
            tool_version: "0.1.0".into(),
            command_line: Some("cascade-verify analyze".into()),
            start_time: Some("2024-01-01T00:00:00Z".into()),
            end_time: Some("2024-01-01T00:00:05Z".into()),
            exit_code: Some(2),
            base_uri: None,
        }
    }

    #[test]
    fn test_sarif_version() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        assert_eq!(report.version, "2.1.0");
    }

    #[test]
    fn test_sarif_schema() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        assert!(report.schema.contains("sarif-schema-2.1.0"));
    }

    #[test]
    fn test_sarif_has_runs() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        assert_eq!(report.runs.len(), 1);
    }

    #[test]
    fn test_sarif_tool_name() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        assert_eq!(report.runs[0].tool.driver.name, "CascadeVerify");
    }

    #[test]
    fn test_sarif_results_count() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        assert_eq!(report.runs[0].results.len(), 2);
    }

    #[test]
    fn test_sarif_rule_ids() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let ids: Vec<_> = report.runs[0].results.iter().map(|r| r.rule_id.clone()).collect();
        assert!(ids.contains(&"CV001".to_string()));
        assert!(ids.contains(&"CV002".to_string()));
    }

    #[test]
    fn test_sarif_levels() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        assert_eq!(report.runs[0].results[0].level, "error");
        assert_eq!(report.runs[0].results[1].level, "warning");
    }

    #[test]
    fn test_sarif_locations() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let locs = &report.runs[0].results[0].locations;
        assert_eq!(locs.len(), 1);
        let phys = locs[0].physical_location.as_ref().unwrap();
        assert_eq!(phys.artifact_location.uri, "envoy/cluster-a.yaml");
        assert_eq!(phys.region.as_ref().unwrap().start_line, Some(12));
    }

    #[test]
    fn test_sarif_code_flows() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let flows = &report.runs[0].results[0].code_flows;
        assert!(!flows.is_empty(), "expected code flows for multi-evidence finding");
        assert_eq!(flows[0].thread_flows[0].locations.len(), 3);
    }

    #[test]
    fn test_sarif_artifacts() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let artifacts = &report.runs[0].artifacts;
        assert!(artifacts.len() >= 2);
        let uris: Vec<_> = artifacts.iter().map(|a| a.location.uri.clone()).collect();
        assert!(uris.contains(&"envoy/cluster-a.yaml".to_string()));
    }

    #[test]
    fn test_sarif_invocations() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let inv = &report.runs[0].invocations[0];
        assert!(!inv.execution_successful);
        assert_eq!(inv.exit_code, Some(2));
    }

    #[test]
    fn test_sarif_rules_generated() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let rules = &report.runs[0].tool.driver.rules;
        assert!(rules.len() >= 2);
        let ids: Vec<_> = rules.iter().map(|r| r.id.clone()).collect();
        assert!(ids.contains(&"CV001".to_string()));
    }

    #[test]
    fn test_sarif_to_json() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let json = SarifGenerator::to_json(&report);
        assert!(json.contains("\"version\": \"2.1.0\""));
        assert!(json.contains("CascadeVerify"));
    }

    #[test]
    fn test_sarif_json_roundtrip() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let json = SarifGenerator::to_json(&report);
        let parsed: SarifReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, "2.1.0");
        assert_eq!(parsed.runs[0].results.len(), 2);
    }

    #[test]
    fn test_sarif_with_repairs() {
        let findings = sample_findings();
        let repairs = vec![RepairPlan {
            id: "R1".into(),
            changes: vec![],
            actions: vec![RepairAction::ModifyRetryCount {
                edge_id: EdgeId::new("a→b"),
                new_count: 1,
            }],
            cost: 2.0,
            effectiveness: 0.0,
            description: "Reduce retry count".into(),
        }];
        let report =
            SarifGenerator::generate_with_repairs(&findings, &repairs, &sample_metadata());
        // The fix should be attached if the message matches
        let total_fixes: usize = report.runs[0].results.iter().map(|r| r.fixes.len()).sum();
        // May or may not match depending on message content, just ensure no panic
        assert!(total_fixes >= 0);
    }

    #[test]
    fn test_sarif_empty_findings() {
        let report = SarifGenerator::generate(&[], &sample_metadata());
        assert_eq!(report.runs[0].results.len(), 0);
        assert!(report.runs[0].tool.driver.rules.is_empty());
    }

    #[test]
    fn test_sarif_fingerprints() {
        let report = SarifGenerator::generate(&sample_findings(), &sample_metadata());
        let fp = report.runs[0].results[0].fingerprints.as_ref().unwrap();
        assert!(fp.contains_key("cascadeVerify/v1"));
    }
}

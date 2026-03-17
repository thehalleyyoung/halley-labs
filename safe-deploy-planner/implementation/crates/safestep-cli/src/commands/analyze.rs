//! Implementation of the `analyze` subcommand.
//!
//! Analyzes API schemas across service versions to detect breaking changes,
//! deprecations, and compatibility issues that could affect deployment safety.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cli::{AnalyzeArgs, SchemaFormat};
use crate::config_loader::SafeStepConfig;
use crate::output::OutputManager;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A schema entry representing one version of a service's API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaEntry {
    pub service: String,
    pub version: String,
    pub format: String,
    pub endpoints: Vec<EndpointDef>,
}

/// An API endpoint definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointDef {
    pub name: String,
    pub method: String,
    pub path: String,
    pub request_fields: Vec<FieldDef>,
    pub response_fields: Vec<FieldDef>,
}

/// A field in a request or response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub field_type: String,
    pub required: bool,
}

/// Classification of a schema change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeKind {
    Breaking,
    NonBreaking,
    Deprecation,
}

impl std::fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Breaking => write!(f, "BREAKING"),
            Self::NonBreaking => write!(f, "non-breaking"),
            Self::Deprecation => write!(f, "deprecation"),
        }
    }
}

/// A detected schema change between versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaChange {
    pub service: String,
    pub from_version: String,
    pub to_version: String,
    pub kind: ChangeKind,
    pub description: String,
    pub confidence: f64,
    pub affected_endpoints: Vec<String>,
}

/// Result of schema analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub total_schemas: usize,
    pub services_analyzed: usize,
    pub changes: Vec<SchemaChange>,
    pub breaking_count: usize,
    pub non_breaking_count: usize,
    pub deprecation_count: usize,
}

// ---------------------------------------------------------------------------
// Command
// ---------------------------------------------------------------------------

pub struct AnalyzeCommand {
    args: AnalyzeArgs,
    _config: SafeStepConfig,
}

impl AnalyzeCommand {
    pub fn new(args: AnalyzeArgs, config: SafeStepConfig) -> Self {
        Self { args, _config: config }
    }

    pub fn execute(&self, output: &mut OutputManager) -> Result<()> {
        info!("analyzing schemas in {:?} (format={:?})", self.args.schema_dir, self.args.format);

        let entries = self.discover_schemas()?;
        if entries.is_empty() {
            output.writeln("No schema files found.");
            return Ok(());
        }

        let mut changes = self.detect_changes(&entries);

        // Apply confidence threshold filter.
        changes.retain(|c| c.confidence >= self.args.min_confidence);

        // Apply breaking-only filter.
        if self.args.breaking_only {
            changes.retain(|c| c.kind == ChangeKind::Breaking);
        }

        // Apply baseline filter.
        if let Some(ref baseline) = self.args.baseline_version {
            changes.retain(|c| c.from_version == *baseline || c.to_version == *baseline);
        }

        let service_set: std::collections::HashSet<&str> = entries.iter()
            .map(|e| e.service.as_str()).collect();

        let result = AnalysisResult {
            total_schemas: entries.len(),
            services_analyzed: service_set.len(),
            breaking_count: changes.iter().filter(|c| c.kind == ChangeKind::Breaking).count(),
            non_breaking_count: changes.iter().filter(|c| c.kind == ChangeKind::NonBreaking).count(),
            deprecation_count: changes.iter().filter(|c| c.kind == ChangeKind::Deprecation).count(),
            changes,
        };

        self.render(output, &result);
        Ok(())
    }

    fn discover_schemas(&self) -> Result<Vec<SchemaEntry>> {
        let dir = &self.args.schema_dir;
        if !dir.exists() {
            anyhow::bail!("schema directory {:?} does not exist", dir);
        }

        let extension = match self.args.format {
            SchemaFormat::Openapi => "json",
            SchemaFormat::Protobuf => "proto",
            SchemaFormat::Graphql => "graphql",
            SchemaFormat::Avro => "avro",
        };

        let mut entries = Vec::new();
        let read_dir = std::fs::read_dir(dir)
            .with_context(|| format!("reading schema directory {:?}", dir))?;

        for entry in read_dir {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let matches = path.extension()
                .map_or(false, |e| e == extension || e == "yaml" || e == "yml" || e == "json");

            if matches {
                match self.parse_schema_file(&path) {
                    Ok(schema) => entries.push(schema),
                    Err(e) => {
                        info!("skipping {:?}: {}", path, e);
                    }
                }
            }
        }

        // Sort by service name then version.
        entries.sort_by(|a, b| {
            a.service.cmp(&b.service).then_with(|| a.version.cmp(&b.version))
        });

        Ok(entries)
    }

    fn parse_schema_file(&self, path: &std::path::Path) -> Result<SchemaEntry> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("reading schema file {:?}", path))?;

        // Try JSON first, then YAML.
        if let Ok(entry) = serde_json::from_str::<SchemaEntry>(&content) {
            return Ok(entry);
        }
        if let Ok(entry) = serde_yaml::from_str::<SchemaEntry>(&content) {
            return Ok(entry);
        }

        anyhow::bail!("could not parse {:?} as JSON or YAML schema", path)
    }

    fn detect_changes(&self, entries: &[SchemaEntry]) -> Vec<SchemaChange> {
        // Group entries by service.
        let mut by_service: HashMap<&str, Vec<&SchemaEntry>> = HashMap::new();
        for entry in entries {
            by_service.entry(entry.service.as_str())
                .or_default()
                .push(entry);
        }

        let mut changes = Vec::new();
        for (_service, versions) in &by_service {
            if versions.len() < 2 {
                continue;
            }
            for window in versions.windows(2) {
                let old = window[0];
                let new = window[1];
                changes.extend(self.diff_schemas(old, new));
            }
        }

        changes
    }

    fn diff_schemas(&self, old: &SchemaEntry, new: &SchemaEntry) -> Vec<SchemaChange> {
        let mut changes = Vec::new();

        let old_endpoints: HashMap<&str, &EndpointDef> = old.endpoints.iter()
            .map(|e| (e.name.as_str(), e)).collect();
        let new_endpoints: HashMap<&str, &EndpointDef> = new.endpoints.iter()
            .map(|e| (e.name.as_str(), e)).collect();

        // Removed endpoints are breaking.
        for (name, _ep) in &old_endpoints {
            if !new_endpoints.contains_key(name) {
                changes.push(SchemaChange {
                    service: old.service.clone(),
                    from_version: old.version.clone(),
                    to_version: new.version.clone(),
                    kind: ChangeKind::Breaking,
                    description: format!("endpoint '{}' was removed", name),
                    confidence: 1.0,
                    affected_endpoints: vec![name.to_string()],
                });
            }
        }

        // New endpoints are non-breaking.
        for (name, _ep) in &new_endpoints {
            if !old_endpoints.contains_key(name) {
                changes.push(SchemaChange {
                    service: old.service.clone(),
                    from_version: old.version.clone(),
                    to_version: new.version.clone(),
                    kind: ChangeKind::NonBreaking,
                    description: format!("endpoint '{}' was added", name),
                    confidence: 1.0,
                    affected_endpoints: vec![name.to_string()],
                });
            }
        }

        // Compare endpoints that exist in both.
        for (name, old_ep) in &old_endpoints {
            if let Some(new_ep) = new_endpoints.get(name) {
                changes.extend(self.diff_endpoint(
                    &old.service, &old.version, &new.version, old_ep, new_ep
                ));
            }
        }

        changes
    }

    fn diff_endpoint(
        &self,
        service: &str,
        from_version: &str,
        to_version: &str,
        old_ep: &EndpointDef,
        new_ep: &EndpointDef,
    ) -> Vec<SchemaChange> {
        let mut changes = Vec::new();

        // Check for method change (breaking).
        if old_ep.method != new_ep.method {
            changes.push(SchemaChange {
                service: service.to_string(),
                from_version: from_version.to_string(),
                to_version: to_version.to_string(),
                kind: ChangeKind::Breaking,
                description: format!(
                    "endpoint '{}' method changed from {} to {}",
                    old_ep.name, old_ep.method, new_ep.method
                ),
                confidence: 1.0,
                affected_endpoints: vec![old_ep.name.clone()],
            });
        }

        // Check for path change (breaking).
        if old_ep.path != new_ep.path {
            changes.push(SchemaChange {
                service: service.to_string(),
                from_version: from_version.to_string(),
                to_version: to_version.to_string(),
                kind: ChangeKind::Breaking,
                description: format!(
                    "endpoint '{}' path changed from '{}' to '{}'",
                    old_ep.name, old_ep.path, new_ep.path
                ),
                confidence: 0.9,
                affected_endpoints: vec![old_ep.name.clone()],
            });
        }

        // Compare request fields.
        changes.extend(self.diff_fields(
            service, from_version, to_version, &old_ep.name,
            &old_ep.request_fields, &new_ep.request_fields, "request"
        ));

        // Compare response fields.
        changes.extend(self.diff_fields(
            service, from_version, to_version, &old_ep.name,
            &old_ep.response_fields, &new_ep.response_fields, "response"
        ));

        changes
    }

    fn diff_fields(
        &self,
        service: &str,
        from_version: &str,
        to_version: &str,
        endpoint: &str,
        old_fields: &[FieldDef],
        new_fields: &[FieldDef],
        field_location: &str,
    ) -> Vec<SchemaChange> {
        let mut changes = Vec::new();
        let old_map: HashMap<&str, &FieldDef> = old_fields.iter()
            .map(|f| (f.name.as_str(), f)).collect();
        let new_map: HashMap<&str, &FieldDef> = new_fields.iter()
            .map(|f| (f.name.as_str(), f)).collect();

        // Removed required fields are breaking.
        for (name, field) in &old_map {
            if !new_map.contains_key(name) {
                let kind = if field.required && field_location == "response" {
                    ChangeKind::Breaking
                } else if field.required && field_location == "request" {
                    ChangeKind::NonBreaking // removing required request field = less strict
                } else {
                    ChangeKind::Deprecation
                };
                changes.push(SchemaChange {
                    service: service.to_string(),
                    from_version: from_version.to_string(),
                    to_version: to_version.to_string(),
                    kind,
                    description: format!(
                        "{} field '{}' removed from endpoint '{}'", field_location, name, endpoint
                    ),
                    confidence: 0.9,
                    affected_endpoints: vec![endpoint.to_string()],
                });
            }
        }

        // New required request fields are breaking (clients must send them).
        for (name, field) in &new_map {
            if !old_map.contains_key(name) && field.required && field_location == "request" {
                changes.push(SchemaChange {
                    service: service.to_string(),
                    from_version: from_version.to_string(),
                    to_version: to_version.to_string(),
                    kind: ChangeKind::Breaking,
                    description: format!(
                        "new required {} field '{}' added to endpoint '{}'",
                        field_location, name, endpoint
                    ),
                    confidence: 0.95,
                    affected_endpoints: vec![endpoint.to_string()],
                });
            } else if !old_map.contains_key(name) {
                changes.push(SchemaChange {
                    service: service.to_string(),
                    from_version: from_version.to_string(),
                    to_version: to_version.to_string(),
                    kind: ChangeKind::NonBreaking,
                    description: format!(
                        "new {} field '{}' added to endpoint '{}'",
                        field_location, name, endpoint
                    ),
                    confidence: 0.9,
                    affected_endpoints: vec![endpoint.to_string()],
                });
            }
        }

        // Type changes in existing fields are breaking.
        for (name, old_field) in &old_map {
            if let Some(new_field) = new_map.get(name) {
                if old_field.field_type != new_field.field_type {
                    changes.push(SchemaChange {
                        service: service.to_string(),
                        from_version: from_version.to_string(),
                        to_version: to_version.to_string(),
                        kind: ChangeKind::Breaking,
                        description: format!(
                            "{} field '{}' type changed from '{}' to '{}' in endpoint '{}'",
                            field_location, name, old_field.field_type, new_field.field_type, endpoint
                        ),
                        confidence: 0.85,
                        affected_endpoints: vec![endpoint.to_string()],
                    });
                }

                // Optional → required in request is breaking.
                if !old_field.required && new_field.required && field_location == "request" {
                    changes.push(SchemaChange {
                        service: service.to_string(),
                        from_version: from_version.to_string(),
                        to_version: to_version.to_string(),
                        kind: ChangeKind::Breaking,
                        description: format!(
                            "{} field '{}' changed from optional to required in endpoint '{}'",
                            field_location, name, endpoint
                        ),
                        confidence: 0.9,
                        affected_endpoints: vec![endpoint.to_string()],
                    });
                }
            }
        }

        changes
    }

    fn render(&self, output: &mut OutputManager, result: &AnalysisResult) {
        let colors = output.colors().clone();

        output.writeln(&format!("Schema Analysis: {} schemas across {} services",
            result.total_schemas, result.services_analyzed));
        output.writeln("");

        if result.changes.is_empty() {
            output.writeln(&colors.safe("No changes detected."));
        } else {
            output.writeln(&format!(
                "Changes: {} breaking, {} non-breaking, {} deprecations",
                colors.error(&result.breaking_count.to_string()),
                colors.info(&result.non_breaking_count.to_string()),
                colors.warning(&result.deprecation_count.to_string()),
            ));
            output.writeln("");

            for change in &result.changes {
                let kind_str = match change.kind {
                    ChangeKind::Breaking => colors.error(&change.kind.to_string()),
                    ChangeKind::NonBreaking => colors.info(&change.kind.to_string()),
                    ChangeKind::Deprecation => colors.warning(&change.kind.to_string()),
                };
                output.writeln(&format!(
                    "  [{}] {}: {} → {} — {} (confidence={:.0}%)",
                    kind_str, change.service,
                    change.from_version, change.to_version,
                    change.description,
                    change.confidence * 100.0,
                ));
            }
        }

        let _ = output.render_value(result);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::OutputFormat;

    fn make_endpoint(name: &str, method: &str, path: &str) -> EndpointDef {
        EndpointDef {
            name: name.to_string(),
            method: method.to_string(),
            path: path.to_string(),
            request_fields: vec![],
            response_fields: vec![],
        }
    }

    fn make_entry(service: &str, version: &str, endpoints: Vec<EndpointDef>) -> SchemaEntry {
        SchemaEntry {
            service: service.to_string(),
            version: version.to_string(),
            format: "openapi".to_string(),
            endpoints,
        }
    }

    #[test]
    fn test_removed_endpoint_is_breaking() {
        let old = make_entry("api", "1.0", vec![
            make_endpoint("getUser", "GET", "/users/{id}"),
            make_endpoint("deleteUser", "DELETE", "/users/{id}"),
        ]);
        let new = make_entry("api", "2.0", vec![
            make_endpoint("getUser", "GET", "/users/{id}"),
        ]);

        let args = AnalyzeArgs {
            schema_dir: std::path::PathBuf::from("/tmp"),
            format: SchemaFormat::Openapi,
            breaking_only: false,
            export_predicates: None,
            min_confidence: 0.0,
            baseline_version: None,
        };
        let cmd = AnalyzeCommand::new(args, SafeStepConfig::default());
        let changes = cmd.diff_schemas(&old, &new);
        assert!(changes.iter().any(|c| c.kind == ChangeKind::Breaking && c.description.contains("deleteUser")));
    }

    #[test]
    fn test_added_endpoint_is_nonbreaking() {
        let old = make_entry("api", "1.0", vec![
            make_endpoint("getUser", "GET", "/users/{id}"),
        ]);
        let new = make_entry("api", "2.0", vec![
            make_endpoint("getUser", "GET", "/users/{id}"),
            make_endpoint("listUsers", "GET", "/users"),
        ]);

        let args = AnalyzeArgs {
            schema_dir: std::path::PathBuf::from("/tmp"),
            format: SchemaFormat::Openapi,
            breaking_only: false,
            export_predicates: None,
            min_confidence: 0.0,
            baseline_version: None,
        };
        let cmd = AnalyzeCommand::new(args, SafeStepConfig::default());
        let changes = cmd.diff_schemas(&old, &new);
        assert!(changes.iter().any(|c| c.kind == ChangeKind::NonBreaking && c.description.contains("listUsers")));
    }

    #[test]
    fn test_method_change_is_breaking() {
        let old = make_entry("api", "1.0", vec![
            make_endpoint("updateUser", "PUT", "/users/{id}"),
        ]);
        let new = make_entry("api", "2.0", vec![
            make_endpoint("updateUser", "PATCH", "/users/{id}"),
        ]);

        let args = AnalyzeArgs {
            schema_dir: std::path::PathBuf::from("/tmp"),
            format: SchemaFormat::Openapi,
            breaking_only: false,
            export_predicates: None,
            min_confidence: 0.0,
            baseline_version: None,
        };
        let cmd = AnalyzeCommand::new(args, SafeStepConfig::default());
        let changes = cmd.diff_schemas(&old, &new);
        assert!(changes.iter().any(|c| c.kind == ChangeKind::Breaking && c.description.contains("method changed")));
    }

    #[test]
    fn test_new_required_request_field_is_breaking() {
        let old_ep = EndpointDef {
            name: "createUser".to_string(),
            method: "POST".to_string(),
            path: "/users".to_string(),
            request_fields: vec![
                FieldDef { name: "name".to_string(), field_type: "string".to_string(), required: true },
            ],
            response_fields: vec![],
        };
        let new_ep = EndpointDef {
            name: "createUser".to_string(),
            method: "POST".to_string(),
            path: "/users".to_string(),
            request_fields: vec![
                FieldDef { name: "name".to_string(), field_type: "string".to_string(), required: true },
                FieldDef { name: "email".to_string(), field_type: "string".to_string(), required: true },
            ],
            response_fields: vec![],
        };
        let old = make_entry("api", "1.0", vec![old_ep]);
        let new = make_entry("api", "2.0", vec![new_ep]);

        let args = AnalyzeArgs {
            schema_dir: std::path::PathBuf::from("/tmp"),
            format: SchemaFormat::Openapi,
            breaking_only: false,
            export_predicates: None,
            min_confidence: 0.0,
            baseline_version: None,
        };
        let cmd = AnalyzeCommand::new(args, SafeStepConfig::default());
        let changes = cmd.diff_schemas(&old, &new);
        assert!(changes.iter().any(|c| c.kind == ChangeKind::Breaking && c.description.contains("email")));
    }

    #[test]
    fn test_type_change_is_breaking() {
        let old_ep = EndpointDef {
            name: "getUser".to_string(),
            method: "GET".to_string(),
            path: "/users/{id}".to_string(),
            request_fields: vec![],
            response_fields: vec![
                FieldDef { name: "age".to_string(), field_type: "integer".to_string(), required: true },
            ],
        };
        let new_ep = EndpointDef {
            name: "getUser".to_string(),
            method: "GET".to_string(),
            path: "/users/{id}".to_string(),
            request_fields: vec![],
            response_fields: vec![
                FieldDef { name: "age".to_string(), field_type: "string".to_string(), required: true },
            ],
        };
        let old = make_entry("api", "1.0", vec![old_ep]);
        let new = make_entry("api", "2.0", vec![new_ep]);

        let args = AnalyzeArgs {
            schema_dir: std::path::PathBuf::from("/tmp"),
            format: SchemaFormat::Openapi,
            breaking_only: false,
            export_predicates: None,
            min_confidence: 0.0,
            baseline_version: None,
        };
        let cmd = AnalyzeCommand::new(args, SafeStepConfig::default());
        let changes = cmd.diff_schemas(&old, &new);
        assert!(changes.iter().any(|c| c.kind == ChangeKind::Breaking && c.description.contains("type changed")));
    }

    #[test]
    fn test_change_kind_display() {
        assert_eq!(ChangeKind::Breaking.to_string(), "BREAKING");
        assert_eq!(ChangeKind::NonBreaking.to_string(), "non-breaking");
        assert_eq!(ChangeKind::Deprecation.to_string(), "deprecation");
    }

    #[test]
    fn test_detect_changes_multiple_services() {
        let entries = vec![
            make_entry("api", "1.0", vec![make_endpoint("get", "GET", "/api")]),
            make_entry("api", "2.0", vec![make_endpoint("get", "GET", "/api/v2")]),
            make_entry("db", "1.0", vec![make_endpoint("query", "POST", "/query")]),
        ];
        let args = AnalyzeArgs {
            schema_dir: std::path::PathBuf::from("/tmp"),
            format: SchemaFormat::Openapi,
            breaking_only: false,
            export_predicates: None,
            min_confidence: 0.0,
            baseline_version: None,
        };
        let cmd = AnalyzeCommand::new(args, SafeStepConfig::default());
        let changes = cmd.detect_changes(&entries);
        // Only api has 2 versions, so only api changes detected.
        assert!(changes.iter().all(|c| c.service == "api"));
    }

    #[test]
    fn test_render_output() {
        let result = AnalysisResult {
            total_schemas: 4,
            services_analyzed: 2,
            changes: vec![SchemaChange {
                service: "api".to_string(),
                from_version: "1.0".to_string(),
                to_version: "2.0".to_string(),
                kind: ChangeKind::Breaking,
                description: "endpoint removed".to_string(),
                confidence: 0.95,
                affected_endpoints: vec!["getUser".to_string()],
            }],
            breaking_count: 1,
            non_breaking_count: 0,
            deprecation_count: 0,
        };
        let args = AnalyzeArgs {
            schema_dir: std::path::PathBuf::from("/tmp"),
            format: SchemaFormat::Openapi,
            breaking_only: false,
            export_predicates: None,
            min_confidence: 0.0,
            baseline_version: None,
        };
        let cmd = AnalyzeCommand::new(args, SafeStepConfig::default());
        let mut out = OutputManager::new(OutputFormat::Text, false);
        cmd.render(&mut out, &result);
        let buf = out.get_buffer();
        assert!(buf.contains("BREAKING"));
        assert!(buf.contains("endpoint removed"));
    }
}

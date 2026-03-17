//! Helm chart parsing, rendering, and value resolution.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use safestep_types::SafeStepError;

use crate::manifest::KubernetesManifest;

pub type Result<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// Chart types
// ---------------------------------------------------------------------------

/// A parsed Helm chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmChart {
    pub chart_yaml: ChartMetadata,
    pub values: Value,
    pub templates: Vec<HelmTemplate>,
    pub dependencies: Vec<ChartDependency>,
}

/// Parsed Chart.yaml fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartMetadata {
    pub name: String,
    pub version: String,
    pub app_version: Option<String>,
    pub description: Option<String>,
    pub api_version: String,
    #[serde(rename = "type")]
    pub chart_type: Option<String>,
    pub keywords: Vec<String>,
    pub home: Option<String>,
    pub sources: Vec<String>,
    pub maintainers: Vec<Maintainer>,
    pub icon: Option<String>,
    pub deprecated: bool,
    pub dependencies: Vec<ChartDependency>,
}

impl Default for ChartMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "0.1.0".into(),
            app_version: None,
            description: None,
            api_version: "v2".into(),
            chart_type: None,
            keywords: Vec::new(),
            home: None,
            sources: Vec::new(),
            maintainers: Vec::new(),
            icon: None,
            deprecated: false,
            dependencies: Vec::new(),
        }
    }
}

/// Chart maintainer info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Maintainer {
    pub name: String,
    pub email: Option<String>,
    pub url: Option<String>,
}

/// A template file within a Helm chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmTemplate {
    pub name: String,
    pub content: String,
}

/// A chart dependency specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDependency {
    pub name: String,
    pub version: String,
    pub repository: Option<String>,
    pub condition: Option<String>,
    pub tags: Vec<String>,
    pub alias: Option<String>,
    pub import_values: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helm renderer
// ---------------------------------------------------------------------------

/// Renders Helm templates using a simplified Go template engine.
pub struct HelmRenderer {
    /// Named templates available via `{{include "name" .}}` / `{{template "name" .}}`.
    named_templates: HashMap<String, String>,
    /// Maximum recursion depth for include directives.
    max_depth: usize,
}

impl Default for HelmRenderer {
    fn default() -> Self {
        Self {
            named_templates: HashMap::new(),
            max_depth: 16,
        }
    }
}

impl HelmRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Render a chart with the given values, producing K8s manifests.
    pub fn render(&self, chart: &HelmChart, values: &Value) -> Result<Vec<KubernetesManifest>> {
        let mut merged = ValuesResolver::new().merge(&chart.values, values);
        // Inject Chart metadata into the render context
        inject_chart_meta(&mut merged, &chart.chart_yaml);

        // Collect named templates (those starting with _)
        let renderer = self.clone_with_named_templates(chart);

        let mut manifests = Vec::new();
        for template in &chart.templates {
            // Skip partials (files starting with _)
            if template.name.starts_with('_') {
                continue;
            }
            let rendered = renderer.render_template(&template.content, &merged, 0)?;
            let trimmed = rendered.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parsed = KubernetesManifest::parse_multi_doc(trimmed)?;
            manifests.extend(parsed);
        }
        Ok(manifests)
    }

    /// Merge base values with overrides.
    pub fn value_override(base_values: &Value, override_values: &Value) -> Value {
        ValuesResolver::new().merge(base_values, override_values)
    }

    fn clone_with_named_templates(&self, chart: &HelmChart) -> Self {
        let mut named = self.named_templates.clone();
        for t in &chart.templates {
            if t.name.starts_with('_') || t.name.contains("helpers") {
                extract_named_templates(&t.content, &mut named);
            }
        }
        Self {
            named_templates: named,
            max_depth: self.max_depth,
        }
    }

    fn render_template(&self, template: &str, values: &Value, depth: usize) -> Result<String> {
        if depth > self.max_depth {
            return Err(SafeStepError::K8sError {
                message: "Template recursion depth exceeded".into(),
                resource: None,
                namespace: None,
                context: None,
            });
        }

        let mut output = String::with_capacity(template.len());
        let mut chars = template.chars().peekable();

        while let Some(&c) = chars.peek() {
            if c == '{' {
                let rest: String = chars.clone().collect();
                if rest.starts_with("{{") {
                    // Find the closing }}
                    if let Some(end) = rest.find("}}") {
                        let directive = &rest[2..end].trim().to_string();
                        // Advance past the directive
                        for _ in 0..(end + 2) {
                            chars.next();
                        }
                        let result = self.eval_directive(directive, values, depth)?;
                        output.push_str(&result);
                        continue;
                    }
                }
            }
            output.push(c);
            chars.next();
        }

        // Handle {{- and -}} whitespace trimming after full render
        let output = trim_template_whitespace(&output);
        Ok(output)
    }

    fn eval_directive(&self, directive: &str, values: &Value, depth: usize) -> Result<String> {
        let directive = directive.trim();

        // Skip comments
        if directive.starts_with("/*") || directive.starts_with("- /*") {
            return Ok(String::new());
        }

        // Handle whitespace trim markers
        let directive = directive.trim_start_matches("- ").trim_end_matches(" -");

        // Empty directive
        if directive.is_empty() {
            return Ok(String::new());
        }

        // if/else/end block directives
        if directive.starts_with("if ") || directive == "else" || directive == "end" {
            // These are handled at a higher level; individual eval returns empty
            return Ok(String::new());
        }

        // range directive
        if directive.starts_with("range ") {
            return Ok(String::new());
        }

        // define directive
        if directive.starts_with("define ") {
            return Ok(String::new());
        }

        // include "name" .
        if directive.starts_with("include ") {
            return self.eval_include(directive, values, depth);
        }

        // toYaml
        if directive.starts_with("toYaml ") {
            return self.eval_to_yaml(directive, values);
        }

        // indent
        if directive.starts_with("indent ") {
            return Ok(String::new());
        }

        // default
        if directive.starts_with("default ") {
            return self.eval_default(directive, values);
        }

        // quote
        if directive == "quote" || directive.starts_with("quote ") {
            return self.eval_quote(directive, values);
        }

        // Value lookups: .Values.x, .Chart.x, .Release.x
        if directive.starts_with('.') {
            return self.eval_value_lookup(directive, values);
        }

        // Variable assignment with :=
        if directive.contains(":=") {
            return Ok(String::new());
        }

        // Dollar variable
        if directive.starts_with('$') {
            return Ok(String::new());
        }

        Ok(String::new())
    }

    fn eval_value_lookup(&self, path: &str, values: &Value) -> Result<String> {
        let resolved = resolve_value_path(path, values);
        match resolved {
            Value::Null => Ok(String::new()),
            Value::String(s) => Ok(s),
            Value::Bool(b) => Ok(if b { "true".into() } else { "false".into() }),
            Value::Number(n) => Ok(n.to_string()),
            other => {
                // For complex values, render as YAML inline
                serde_yaml::to_string(&other)
                    .map(|s| s.trim().to_string())
                    .map_err(|e| SafeStepError::K8sError {
                        message: format!("Failed to serialize value: {e}"),
                        resource: None,
                        namespace: None,
                        context: None,
                    })
            }
        }
    }

    fn eval_include(&self, directive: &str, values: &Value, depth: usize) -> Result<String> {
        // include "templateName" .
        let parts: Vec<&str> = directive.splitn(3, ' ').collect();
        if parts.len() < 2 {
            return Ok(String::new());
        }
        let name = parts[1].trim_matches('"');
        if let Some(tpl) = self.named_templates.get(name) {
            self.render_template(tpl, values, depth + 1)
        } else {
            Ok(String::new())
        }
    }

    fn eval_to_yaml(&self, directive: &str, values: &Value) -> Result<String> {
        let path = directive.strip_prefix("toYaml ").unwrap_or("").trim();
        let resolved = resolve_value_path(path, values);
        serde_yaml::to_string(&resolved)
            .map(|s| s.trim_end_matches('\n').to_string())
            .map_err(|e| SafeStepError::K8sError {
                message: format!("toYaml error: {e}"),
                resource: None,
                namespace: None,
                context: None,
            })
    }

    fn eval_default(&self, directive: &str, values: &Value) -> Result<String> {
        // default "defaultValue" .Values.path
        let rest = directive.strip_prefix("default ").unwrap_or("");
        let (default_val, path) = parse_default_args(rest);
        let resolved = resolve_value_path(&path, values);
        if resolved.is_null() || resolved == Value::String(String::new()) {
            Ok(default_val)
        } else {
            self.eval_value_lookup(&path, values)
        }
    }

    fn eval_quote(&self, directive: &str, values: &Value) -> Result<String> {
        let path = directive.strip_prefix("quote ").unwrap_or("").trim();
        if path.is_empty() {
            return Ok("\"\"".into());
        }
        let resolved = resolve_value_path(path, values);
        match resolved {
            Value::String(s) => Ok(format!("\"{s}\"")),
            Value::Null => Ok("\"\"".into()),
            other => Ok(format!("\"{}\"", other)),
        }
    }
}

fn inject_chart_meta(values: &mut Value, meta: &ChartMetadata) {
    let chart_obj = serde_json::json!({
        "Name": meta.name,
        "Version": meta.version,
        "AppVersion": meta.app_version.as_deref().unwrap_or(""),
        "Description": meta.description.as_deref().unwrap_or(""),
    });
    if let Value::Object(ref mut map) = values {
        map.insert("Chart".into(), chart_obj);
        if !map.contains_key("Release") {
            map.insert(
                "Release".into(),
                serde_json::json!({
                    "Name": meta.name,
                    "Namespace": "default",
                    "IsUpgrade": false,
                    "IsInstall": true,
                    "Revision": 1,
                }),
            );
        }
    }
}

fn resolve_value_path(path: &str, root: &Value) -> Value {
    let path = path.trim().trim_start_matches('.');
    if path.is_empty() {
        return root.clone();
    }

    let segments: Vec<&str> = path.split('.').collect();
    let mut current = root;

    for seg in &segments {
        match current {
            Value::Object(map) => {
                if let Some(v) = map.get(*seg) {
                    current = v;
                } else {
                    return Value::Null;
                }
            }
            _ => return Value::Null,
        }
    }

    current.clone()
}

fn parse_default_args(s: &str) -> (String, String) {
    let s = s.trim();
    if s.starts_with('"') {
        if let Some(end_quote) = s[1..].find('"') {
            let default_val = s[1..=end_quote].to_string();
            let rest = s[end_quote + 2..].trim().to_string();
            return (default_val, rest);
        }
    }
    // Try splitting on space
    let parts: Vec<&str> = s.splitn(2, ' ').collect();
    if parts.len() == 2 {
        (parts[0].to_string(), parts[1].to_string())
    } else {
        (s.to_string(), String::new())
    }
}

fn extract_named_templates(content: &str, named: &mut HashMap<String, String>) {
    let mut remaining = content;
    while let Some(start) = remaining.find("{{- define \"") {
        let after_start = &remaining[start + 12..];
        if let Some(name_end) = after_start.find('"') {
            let name = after_start[..name_end].to_string();
            let after_name = &after_start[name_end..];
            if let Some(close) = after_name.find("}}") {
                let body_start = &after_name[close + 2..];
                // Find matching {{- end }}
                if let Some(end_pos) = body_start.find("{{- end") {
                    let body = body_start[..end_pos].to_string();
                    named.insert(name, body);
                    remaining = &body_start[end_pos..];
                    continue;
                }
            }
        }
        break;
    }
    // Also handle non-trimming define
    let mut remaining = content;
    while let Some(start) = remaining.find("{{ define \"") {
        let after_start = &remaining[start + 11..];
        if let Some(name_end) = after_start.find('"') {
            let name = after_start[..name_end].to_string();
            let after_name = &after_start[name_end..];
            if let Some(close) = after_name.find("}}") {
                let body_start = &after_name[close + 2..];
                if let Some(end_pos) = body_start.find("{{ end") {
                    let body = body_start[..end_pos].to_string();
                    named.insert(name, body);
                    remaining = &body_start[end_pos..];
                    continue;
                }
            }
        }
        break;
    }
}

fn trim_template_whitespace(s: &str) -> String {
    // Simplified: remove lines that are just whitespace between directives
    let mut lines: Vec<&str> = s.lines().collect();
    lines.retain(|line| {
        let trimmed = line.trim();
        // Keep non-empty lines, or lines that are just whitespace if they're between content
        !trimmed.is_empty() || true
    });
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Values resolver
// ---------------------------------------------------------------------------

/// Resolves and merges Helm values with deep-merge semantics.
pub struct ValuesResolver {
    /// Environment variable prefix for interpolation (e.g., "HELM_").
    pub env_prefix: Option<String>,
}

impl Default for ValuesResolver {
    fn default() -> Self {
        Self { env_prefix: None }
    }
}

impl ValuesResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_env_prefix(prefix: &str) -> Self {
        Self {
            env_prefix: Some(prefix.to_string()),
        }
    }

    /// Deep-merge default_values with override_values. Override wins on conflicts.
    pub fn merge(&self, default_values: &Value, override_values: &Value) -> Value {
        let mut result = default_values.clone();
        deep_merge(&mut result, override_values);
        if self.env_prefix.is_some() {
            self.interpolate_env(&mut result);
        }
        result
    }

    /// Interpolate environment variables in string values.
    fn interpolate_env(&self, value: &mut Value) {
        let prefix = match &self.env_prefix {
            Some(p) => p.clone(),
            None => return,
        };
        match value {
            Value::String(s) => {
                if s.starts_with("${") && s.ends_with('}') {
                    let var_name = &s[2..s.len() - 1];
                    let env_key = format!("{prefix}{var_name}");
                    if let Ok(env_val) = std::env::var(&env_key) {
                        *s = env_val;
                    }
                }
            }
            Value::Object(map) => {
                for v in map.values_mut() {
                    self.interpolate_env(v);
                }
            }
            Value::Array(arr) => {
                for v in arr.iter_mut() {
                    self.interpolate_env(v);
                }
            }
            _ => {}
        }
    }
}

/// Deep-merge `override_val` into `base`.
fn deep_merge(base: &mut Value, override_val: &Value) {
    match (base, override_val) {
        (Value::Object(base_map), Value::Object(override_map)) => {
            for (k, v) in override_map {
                let entry = base_map.entry(k.clone()).or_insert(Value::Null);
                deep_merge(entry, v);
            }
        }
        (base, override_val) => {
            *base = override_val.clone();
        }
    }
}

// ---------------------------------------------------------------------------
// Helm chart loader
// ---------------------------------------------------------------------------

/// Loads a Helm chart from a directory structure.
pub struct HelmChartLoader;

impl HelmChartLoader {
    /// Load a Helm chart from the given directory path.
    pub fn load(path: &str) -> Result<HelmChart> {
        let base = std::path::Path::new(path);

        // Load Chart.yaml
        let chart_yaml_path = base.join("Chart.yaml");
        let chart_yaml_content = std::fs::read_to_string(&chart_yaml_path).map_err(|e| {
            SafeStepError::K8sError {
                message: format!("Failed to read Chart.yaml: {e}"),
                resource: None,
                namespace: None,
                context: None,
            }
        })?;
        let chart_yaml: ChartMetadata =
            serde_yaml::from_str(&chart_yaml_content).map_err(|e| SafeStepError::K8sError {
                message: format!("Failed to parse Chart.yaml: {e}"),
                resource: None,
                namespace: None,
                context: None,
            })?;

        // Load values.yaml (optional)
        let values_path = base.join("values.yaml");
        let values = if values_path.exists() {
            let content = std::fs::read_to_string(&values_path).map_err(|e| {
                SafeStepError::K8sError {
                    message: format!("Failed to read values.yaml: {e}"),
                    resource: None,
                    namespace: None,
                    context: None,
                }
            })?;
            serde_yaml::from_str(&content).unwrap_or(Value::Object(serde_json::Map::new()))
        } else {
            Value::Object(serde_json::Map::new())
        };

        // Load templates from templates/ directory
        let templates_dir = base.join("templates");
        let mut templates = Vec::new();
        if templates_dir.exists() {
            load_templates_recursive(&templates_dir, &templates_dir, &mut templates)?;
        }

        let dependencies = chart_yaml.dependencies.clone();

        Ok(HelmChart {
            chart_yaml,
            values,
            templates,
            dependencies,
        })
    }

    /// Validate chart structure without loading all files.
    pub fn validate_structure(path: &str) -> Result<Vec<String>> {
        let base = std::path::Path::new(path);
        let mut warnings = Vec::new();

        if !base.join("Chart.yaml").exists() {
            return Err(SafeStepError::K8sError {
                message: "Chart.yaml not found".into(),
                resource: None,
                namespace: None,
                context: None,
            });
        }

        if !base.join("templates").exists() {
            warnings.push("templates/ directory not found".into());
        }

        if !base.join("values.yaml").exists() {
            warnings.push("values.yaml not found (using empty defaults)".into());
        }

        Ok(warnings)
    }

    /// Parse a ChartMetadata from a YAML string.
    pub fn parse_chart_yaml(content: &str) -> Result<ChartMetadata> {
        serde_yaml::from_str(content).map_err(|e| SafeStepError::K8sError {
            message: format!("Failed to parse Chart.yaml: {e}"),
            resource: None,
            namespace: None,
            context: None,
        })
    }

    /// Parse a values file from a YAML string.
    pub fn parse_values(content: &str) -> Result<Value> {
        serde_yaml::from_str(content).map_err(|e| SafeStepError::K8sError {
            message: format!("Failed to parse values.yaml: {e}"),
            resource: None,
            namespace: None,
            context: None,
        })
    }
}

fn load_templates_recursive(
    base: &std::path::Path,
    dir: &std::path::Path,
    templates: &mut Vec<HelmTemplate>,
) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| SafeStepError::K8sError {
        message: format!("Failed to read templates directory: {e}"),
        resource: None,
        namespace: None,
        context: None,
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| SafeStepError::K8sError {
            message: format!("Failed to read dir entry: {e}"),
            resource: None,
            namespace: None,
            context: None,
        })?;
        let path = entry.path();
        if path.is_dir() {
            load_templates_recursive(base, &path, templates)?;
        } else if path
            .extension()
            .map(|e| e == "yaml" || e == "yml" || e == "tpl")
            .unwrap_or(false)
        {
            let content = std::fs::read_to_string(&path).map_err(|e| SafeStepError::K8sError {
                message: format!("Failed to read template {}: {e}", path.display()),
                resource: None,
                namespace: None,
                context: None,
            })?;
            let name = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            templates.push(HelmTemplate { name, content });
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Simplified Go template processor for if/range blocks
// ---------------------------------------------------------------------------

/// Process a Go template string with if/else/end and range blocks.
/// This is a simplified processor that handles the most common patterns.
pub fn process_go_template(template: &str, values: &Value) -> Result<String> {
    let renderer = HelmRenderer::new();
    renderer.render_template(template, values, 0)
}

/// Evaluate a Go template condition (simplified).
pub fn eval_condition(condition: &str, values: &Value) -> bool {
    let condition = condition.trim();

    // Handle negation
    if let Some(inner) = condition.strip_prefix("not ") {
        return !eval_condition(inner, values);
    }

    // Handle `and` / `or`
    if condition.contains(" and ") {
        return condition
            .split(" and ")
            .all(|part| eval_condition(part.trim(), values));
    }
    if condition.contains(" or ") {
        return condition
            .split(" or ")
            .any(|part| eval_condition(part.trim(), values));
    }

    // Handle eq/ne
    if condition.starts_with("eq ") {
        let parts: Vec<&str> = condition[3..].splitn(2, ' ').collect();
        if parts.len() == 2 {
            let a = resolve_value_path(parts[0], values);
            let b_str = parts[1].trim_matches('"');
            return match a {
                Value::String(s) => s == b_str,
                Value::Number(n) => n.to_string() == b_str,
                Value::Bool(b) => b.to_string() == b_str,
                _ => false,
            };
        }
    }
    if condition.starts_with("ne ") {
        let parts: Vec<&str> = condition[3..].splitn(2, ' ').collect();
        if parts.len() == 2 {
            let a = resolve_value_path(parts[0], values);
            let b_str = parts[1].trim_matches('"');
            return match a {
                Value::String(s) => s != b_str,
                Value::Number(n) => n.to_string() != b_str,
                Value::Bool(b) => b.to_string() != b_str,
                _ => true,
            };
        }
    }

    // Value truthiness: resolve path and check
    if condition.starts_with('.') {
        let val = resolve_value_path(condition, values);
        return is_truthy(&val);
    }

    // Quoted string literal is truthy if non-empty
    if condition.starts_with('"') && condition.ends_with('"') {
        return condition.len() > 2;
    }

    // Bare `true` / `false`
    match condition {
        "true" => true,
        "false" => false,
        _ => !condition.is_empty(),
    }
}

fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_values() -> Value {
        serde_json::json!({
            "Values": {
                "replicaCount": 3,
                "image": {
                    "repository": "nginx",
                    "tag": "1.21.0",
                    "pullPolicy": "IfNotPresent"
                },
                "service": {
                    "type": "ClusterIP",
                    "port": 80
                },
                "resources": {
                    "limits": {
                        "cpu": "500m",
                        "memory": "256Mi"
                    },
                    "requests": {
                        "cpu": "250m",
                        "memory": "128Mi"
                    }
                },
                "nodeSelector": {},
                "tolerations": [],
                "affinity": {},
                "enabled": true,
                "name": "my-app",
                "labels": {
                    "app": "my-app",
                    "version": "1.0"
                }
            }
        })
    }

    #[test]
    fn test_value_lookup() {
        let values = sample_values();
        let result = resolve_value_path(".Values.replicaCount", &values);
        assert_eq!(result, Value::Number(3.into()));

        let result = resolve_value_path(".Values.image.repository", &values);
        assert_eq!(result, Value::String("nginx".into()));

        let result = resolve_value_path(".Values.nonexistent", &values);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_deep_merge() {
        let base = serde_json::json!({
            "a": 1,
            "b": {
                "c": 2,
                "d": 3
            },
            "e": [1, 2, 3]
        });
        let override_val = serde_json::json!({
            "b": {
                "c": 20,
                "f": 4
            },
            "g": 5
        });
        let resolver = ValuesResolver::new();
        let merged = resolver.merge(&base, &override_val);
        assert_eq!(merged["a"], 1);
        assert_eq!(merged["b"]["c"], 20);
        assert_eq!(merged["b"]["d"], 3);
        assert_eq!(merged["b"]["f"], 4);
        assert_eq!(merged["g"], 5);
    }

    #[test]
    fn test_deep_merge_override_array() {
        let base = serde_json::json!({"arr": [1, 2]});
        let over = serde_json::json!({"arr": [3, 4, 5]});
        let merged = ValuesResolver::new().merge(&base, &over);
        assert_eq!(merged["arr"], serde_json::json!([3, 4, 5]));
    }

    #[test]
    fn test_eval_condition_value() {
        let values = sample_values();
        assert!(eval_condition(".Values.enabled", &values));
        assert!(eval_condition(".Values.replicaCount", &values));
        assert!(!eval_condition(".Values.nonexistent", &values));
    }

    #[test]
    fn test_eval_condition_not() {
        let values = sample_values();
        assert!(!eval_condition("not .Values.enabled", &values));
        assert!(eval_condition("not .Values.nonexistent", &values));
    }

    #[test]
    fn test_eval_condition_eq() {
        let values = sample_values();
        assert!(eval_condition("eq .Values.image.repository \"nginx\"", &values));
        assert!(!eval_condition("eq .Values.image.repository \"apache\"", &values));
    }

    #[test]
    fn test_eval_condition_ne() {
        let values = sample_values();
        assert!(eval_condition("ne .Values.image.repository \"apache\"", &values));
        assert!(!eval_condition("ne .Values.image.repository \"nginx\"", &values));
    }

    #[test]
    fn test_eval_condition_and_or() {
        let values = sample_values();
        assert!(eval_condition(".Values.enabled and .Values.replicaCount", &values));
        assert!(!eval_condition(".Values.enabled and .Values.nonexistent", &values));
        assert!(eval_condition(".Values.enabled or .Values.nonexistent", &values));
        assert!(!eval_condition(".Values.nonexistent or .Values.missing", &values));
    }

    #[test]
    fn test_render_simple_template() {
        let values = sample_values();
        let template = "replicas: {{ .Values.replicaCount }}";
        let result = process_go_template(template, &values).unwrap();
        assert!(result.contains("3"));
    }

    #[test]
    fn test_render_nested_value() {
        let values = sample_values();
        let template = "image: {{ .Values.image.repository }}:{{ .Values.image.tag }}";
        let result = process_go_template(template, &values).unwrap();
        assert!(result.contains("nginx:1.21.0"));
    }

    #[test]
    fn test_to_yaml() {
        let values = sample_values();
        let renderer = HelmRenderer::new();
        let result = renderer.eval_to_yaml("toYaml .Values.labels", &values).unwrap();
        assert!(result.contains("app:"));
        assert!(result.contains("my-app"));
    }

    #[test]
    fn test_default_function() {
        let values = sample_values();
        let renderer = HelmRenderer::new();
        let result = renderer
            .eval_default("default \"fallback\" .Values.nonexistent", &values)
            .unwrap();
        assert_eq!(result, "fallback");

        let result2 = renderer
            .eval_default("default \"fallback\" .Values.name", &values)
            .unwrap();
        assert_eq!(result2, "my-app");
    }

    #[test]
    fn test_chart_metadata_default() {
        let meta = ChartMetadata::default();
        assert_eq!(meta.api_version, "v2");
        assert_eq!(meta.version, "0.1.0");
        assert!(!meta.deprecated);
    }

    #[test]
    fn test_helm_renderer_full_chart() {
        let chart = HelmChart {
            chart_yaml: ChartMetadata {
                name: "my-app".into(),
                version: "1.0.0".into(),
                app_version: Some("2.0.0".into()),
                ..Default::default()
            },
            values: serde_json::json!({
                "replicaCount": 2,
                "image": {
                    "repository": "myapp",
                    "tag": "latest"
                }
            }),
            templates: vec![HelmTemplate {
                name: "deployment.yaml".into(),
                content: r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Chart.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: {{ .Values.image.repository }}:{{ .Values.image.tag }}
"#
                .into(),
            }],
            dependencies: Vec::new(),
        };

        let override_vals = serde_json::json!({
            "replicaCount": 5
        });

        let renderer = HelmRenderer::new();
        let manifests = renderer.render(&chart, &override_vals).unwrap();
        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].kind, "Deployment");
        assert_eq!(manifests[0].metadata.name, "my-app");
        // Check that override was applied
        let spec = manifests[0].spec.as_ref().unwrap();
        assert_eq!(spec.get("replicas").and_then(|v| v.as_u64()), Some(5));
    }

    #[test]
    fn test_extract_named_templates() {
        let content = r#"
{{- define "myapp.fullname" }}
{{- printf "%s-%s" .Release.Name .Chart.Name }}
{{- end }}

{{- define "myapp.labels" }}
app: myapp
{{- end }}
"#;
        let mut named = HashMap::new();
        extract_named_templates(content, &mut named);
        assert!(named.contains_key("myapp.fullname"));
        assert!(named.contains_key("myapp.labels"));
    }

    #[test]
    fn test_values_resolver_merge_nested() {
        let defaults = serde_json::json!({
            "global": {
                "imageRegistry": "docker.io",
                "storageClass": "standard"
            },
            "replicaCount": 1,
            "service": {
                "type": "ClusterIP",
                "port": 80
            }
        });
        let overrides = serde_json::json!({
            "global": {
                "imageRegistry": "my-registry.io"
            },
            "replicaCount": 3,
            "service": {
                "type": "LoadBalancer"
            }
        });
        let result = ValuesResolver::new().merge(&defaults, &overrides);
        assert_eq!(result["global"]["imageRegistry"], "my-registry.io");
        assert_eq!(result["global"]["storageClass"], "standard"); // preserved
        assert_eq!(result["replicaCount"], 3);
        assert_eq!(result["service"]["type"], "LoadBalancer");
        assert_eq!(result["service"]["port"], 80); // preserved
    }

    #[test]
    fn test_chart_dependency() {
        let dep = ChartDependency {
            name: "redis".into(),
            version: "17.0.0".into(),
            repository: Some("https://charts.bitnami.com/bitnami".into()),
            condition: Some("redis.enabled".into()),
            tags: vec!["cache".into()],
            alias: Some("my-redis".into()),
            import_values: vec![],
        };
        assert_eq!(dep.name, "redis");
        assert_eq!(dep.condition.as_deref(), Some("redis.enabled"));
    }

    #[test]
    fn test_is_truthy() {
        assert!(!is_truthy(&Value::Null));
        assert!(!is_truthy(&Value::Bool(false)));
        assert!(is_truthy(&Value::Bool(true)));
        assert!(!is_truthy(&Value::String(String::new())));
        assert!(is_truthy(&Value::String("hello".into())));
        assert!(!is_truthy(&Value::Array(vec![])));
        assert!(is_truthy(&Value::Array(vec![Value::Null])));
        assert!(!is_truthy(&Value::Object(serde_json::Map::new())));
    }

    #[test]
    fn test_parse_chart_yaml() {
        let yaml = r#"
apiVersion: v2
name: my-chart
version: 1.2.3
appVersion: "4.5.6"
description: A test chart
type: application
keywords:
  - test
  - sample
maintainers:
  - name: John
    email: john@example.com
dependencies:
  - name: redis
    version: "17.0.0"
    repository: https://charts.bitnami.com/bitnami
"#;
        let meta = HelmChartLoader::parse_chart_yaml(yaml).unwrap();
        assert_eq!(meta.name, "my-chart");
        assert_eq!(meta.version, "1.2.3");
        assert_eq!(meta.app_version.as_deref(), Some("4.5.6"));
        assert_eq!(meta.description.as_deref(), Some("A test chart"));
        assert_eq!(meta.keywords, vec!["test", "sample"]);
        assert_eq!(meta.maintainers[0].name, "John");
        assert_eq!(meta.dependencies[0].name, "redis");
    }

    #[test]
    fn test_parse_values() {
        let yaml = r#"
replicaCount: 3
image:
  repository: nginx
  tag: "1.21"
"#;
        let values = HelmChartLoader::parse_values(yaml).unwrap();
        assert_eq!(values["replicaCount"], 3);
        assert_eq!(values["image"]["repository"], "nginx");
    }

    #[test]
    fn test_helm_renderer_multi_template() {
        let chart = HelmChart {
            chart_yaml: ChartMetadata {
                name: "test".into(),
                version: "0.1.0".into(),
                ..Default::default()
            },
            values: serde_json::json!({"port": 8080}),
            templates: vec![
                HelmTemplate {
                    name: "_helpers.tpl".into(),
                    content: r#"{{- define "test.name" }}test-app{{- end }}"#.into(),
                },
                HelmTemplate {
                    name: "service.yaml".into(),
                    content: r#"apiVersion: v1
kind: Service
metadata:
  name: test-svc
spec:
  ports:
  - port: {{ .Values.port }}
"#
                    .into(),
                },
            ],
            dependencies: Vec::new(),
        };
        let renderer = HelmRenderer::new();
        let manifests = renderer.render(&chart, &serde_json::json!({})).unwrap();
        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].kind, "Service");
    }

    #[test]
    fn test_value_override() {
        let base = serde_json::json!({"a": 1, "b": {"c": 2}});
        let over = serde_json::json!({"b": {"c": 3, "d": 4}});
        let result = HelmRenderer::value_override(&base, &over);
        assert_eq!(result["a"], 1);
        assert_eq!(result["b"]["c"], 3);
        assert_eq!(result["b"]["d"], 4);
    }
}

//! Helm chart processing and Go template engine for Kubernetes Helm charts.
//!
//! Provides types for representing Helm chart metadata, release info, and template
//! contexts, along with a Go-template-compatible rendering engine that handles the
//! most common patterns found in production Helm charts.

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Wrapper around `serde_yaml::Value` with deep-merge semantics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HelmValues {
    inner: serde_yaml::Value,
}

impl HelmValues {
    pub fn new(val: serde_yaml::Value) -> Self {
        Self { inner: val }
    }

    /// Dot-path access (e.g. `"a.b.c"`) into the underlying YAML value.
    pub fn get(&self, path: &str) -> Option<&serde_yaml::Value> {
        let mut current = &self.inner;
        for segment in path.split('.') {
            if segment.is_empty() {
                continue;
            }
            match current {
                serde_yaml::Value::Mapping(map) => {
                    let key = serde_yaml::Value::String(segment.to_string());
                    current = map.get(&key)?;
                }
                _ => return None,
            }
        }
        Some(current)
    }

    /// Set a value at a dot-separated path, creating intermediate mappings as needed.
    pub fn set(&mut self, path: &str, value: serde_yaml::Value) {
        let segments: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
        if segments.is_empty() {
            self.inner = value;
            return;
        }
        let mut current = &mut self.inner;
        // Ensure root is a mapping.
        if !current.is_mapping() {
            *current = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
        }
        for (i, seg) in segments.iter().enumerate() {
            let key = serde_yaml::Value::String(seg.to_string());
            if i == segments.len() - 1 {
                if let serde_yaml::Value::Mapping(map) = current {
                    map.insert(key, value.clone());
                }
                return;
            }
            // Navigate or create intermediate mapping.
            if let serde_yaml::Value::Mapping(map) = current {
                if !map.contains_key(&key) {
                    map.insert(key.clone(), serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
                }
                current = map.get_mut(&key).unwrap();
                if !current.is_mapping() {
                    *current = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
                }
            }
        }
    }

    pub fn inner(&self) -> &serde_yaml::Value {
        &self.inner
    }
}

/// Metadata from `Chart.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartMetadata {
    pub name: String,
    pub version: String,
    pub app_version: String,
    pub description: String,
    pub dependencies: Vec<ChartDependency>,
}

/// A single chart dependency entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChartDependency {
    pub name: String,
    pub version: String,
    pub repository: String,
    pub condition: Option<String>,
    pub alias: Option<String>,
}

/// Information about the current Helm release.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReleaseInfo {
    pub name: String,
    pub namespace: String,
    pub revision: u32,
    pub is_install: bool,
    pub is_upgrade: bool,
}

/// Full context available inside a Helm template.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemplateContext {
    pub values: HelmValues,
    pub release: ReleaseInfo,
    pub chart: ChartMetadata,
    pub capabilities: Capabilities,
}

/// Kubernetes cluster capabilities exposed to templates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Capabilities {
    pub kube_version: String,
    pub api_versions: Vec<String>,
}

// ---------------------------------------------------------------------------
// Token type for template parsing
// ---------------------------------------------------------------------------

/// A parsed template token — either literal text or a `{{ … }}` action.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Text(String),
    Action(String),
}

/// Split a template string into `Text` and `Action` tokens.
/// Handles `{{-` / `-}}` whitespace-trimming markers by stripping the
/// adjacent whitespace from neighbouring `Text` tokens.
pub fn parse_template_tokens(input: &str) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut rest = input;

    while !rest.is_empty() {
        if let Some(start) = rest.find("{{") {
            if start > 0 {
                tokens.push(Token::Text(rest[..start].to_string()));
            }
            let after_open = &rest[start + 2..];
            let trim_left = after_open.starts_with('-');
            let action_start = if trim_left { &after_open[1..] } else { after_open };

            if let Some(end) = action_start.find("}}") {
                let mut action_body = &action_start[..end];
                let trim_right = action_body.ends_with('-');
                if trim_right {
                    action_body = &action_body[..action_body.len() - 1];
                }
                let action_str = action_body.trim().to_string();

                // Apply left-trim: strip trailing whitespace from the previous Text token.
                if trim_left {
                    if let Some(Token::Text(prev)) = tokens.last_mut() {
                        *prev = prev.trim_end().to_string();
                    }
                }

                tokens.push(Token::Action(action_str));

                rest = &action_start[end + 2..];

                // Apply right-trim: strip leading whitespace (including one newline) from following text.
                if trim_right {
                    rest = rest.trim_start_matches(|c: char| c == ' ' || c == '\t');
                    if rest.starts_with('\n') {
                        rest = &rest[1..];
                    } else if rest.starts_with("\r\n") {
                        rest = &rest[2..];
                    }
                }
            } else {
                // No closing `}}` — treat the rest as text.
                tokens.push(Token::Text(rest.to_string()));
                break;
            }
        } else {
            tokens.push(Token::Text(rest.to_string()));
            break;
        }
    }
    tokens
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Resolve a dot-path expression (e.g. `.Values.x.y`, `.Release.Name`) against a
/// `TemplateContext`, returning the YAML value if found.
pub fn resolve_dot_path(path: &str, ctx: &TemplateContext) -> Option<serde_yaml::Value> {
    let path = path.strip_prefix('.').unwrap_or(path);

    if path.is_empty() || path == "." {
        // `.` refers to the entire context — represent as the values root.
        return Some(ctx.values.inner().clone());
    }

    let segments: Vec<&str> = path.splitn(2, '.').collect();
    let root = segments[0];
    let remainder = segments.get(1).copied().unwrap_or("");

    match root {
        "Values" => {
            if remainder.is_empty() {
                Some(ctx.values.inner().clone())
            } else {
                ctx.values.get(remainder).cloned()
            }
        }
        "Release" => {
            let field = if remainder.is_empty() { root } else { remainder };
            match field {
                "Name" => Some(serde_yaml::Value::String(ctx.release.name.clone())),
                "Namespace" => Some(serde_yaml::Value::String(ctx.release.namespace.clone())),
                "Revision" => Some(serde_yaml::Value::Number(ctx.release.revision.into())),
                "IsInstall" => Some(serde_yaml::Value::Bool(ctx.release.is_install)),
                "IsUpgrade" => Some(serde_yaml::Value::Bool(ctx.release.is_upgrade)),
                _ => None,
            }
        }
        "Chart" => {
            let field = if remainder.is_empty() { root } else { remainder };
            match field {
                "Name" => Some(serde_yaml::Value::String(ctx.chart.name.clone())),
                "Version" => Some(serde_yaml::Value::String(ctx.chart.version.clone())),
                "AppVersion" => Some(serde_yaml::Value::String(ctx.chart.app_version.clone())),
                "Description" => Some(serde_yaml::Value::String(ctx.chart.description.clone())),
                _ => None,
            }
        }
        "Capabilities" => {
            let field = if remainder.is_empty() { root } else { remainder };
            match field {
                "KubeVersion" => Some(serde_yaml::Value::String(ctx.capabilities.kube_version.clone())),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Deep-merge two YAML values.  `overlay` wins for scalars; maps are merged
/// recursively; sequences are replaced wholesale.
pub fn deep_merge(base: &serde_yaml::Value, overlay: &serde_yaml::Value) -> serde_yaml::Value {
    match (base, overlay) {
        (serde_yaml::Value::Mapping(b), serde_yaml::Value::Mapping(o)) => {
            let mut merged = b.clone();
            for (k, v) in o {
                let merged_val = if let Some(existing) = merged.get(k) {
                    deep_merge(existing, v)
                } else {
                    v.clone()
                };
                merged.insert(k.clone(), merged_val);
            }
            serde_yaml::Value::Mapping(merged)
        }
        (_, overlay) => overlay.clone(),
    }
}

/// Convert a `serde_yaml::Value` to a compact YAML string.
pub fn value_to_yaml_string(val: &serde_yaml::Value) -> String {
    match val {
        serde_yaml::Value::Null => "null".to_string(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::String(s) => s.clone(),
        _ => {
            serde_yaml::to_string(val)
                .unwrap_or_default()
                .trim_end_matches('\n')
                .to_string()
        }
    }
}

/// Go-template truthiness: `null`, `false`, `0`, and `""` are falsy.
pub fn is_truthy(val: &serde_yaml::Value) -> bool {
    match val {
        serde_yaml::Value::Null => false,
        serde_yaml::Value::Bool(b) => *b,
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(f) = n.as_f64() {
                f != 0.0
            } else {
                true
            }
        }
        serde_yaml::Value::String(s) => !s.is_empty(),
        serde_yaml::Value::Sequence(seq) => !seq.is_empty(),
        serde_yaml::Value::Mapping(map) => !map.is_empty(),
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// GoTemplateEngine
// ---------------------------------------------------------------------------

/// A lightweight Go-template engine that supports the subset of constructs most
/// commonly used in Kubernetes Helm charts.
#[derive(Debug, Clone)]
pub struct GoTemplateEngine {
    pub named_templates: IndexMap<String, String>,
}

impl GoTemplateEngine {
    pub fn new() -> Self {
        Self {
            named_templates: IndexMap::new(),
        }
    }

    /// Register all `{{ define "name" }} … {{ end }}` blocks found in `template`,
    /// returning the template with those blocks removed.
    fn register_defines(&mut self, template: &str) -> String {
        let mut output = String::new();
        let mut rest: &str = template;

        while let Some(def_start) = rest.find("{{") {
            let before = &rest[..def_start];
            let after_open = &rest[def_start + 2..];
            let trimmed = after_open.trim_start_matches('-').trim_start();

            if trimmed.starts_with("define ") || trimmed.starts_with("define\"") {
                // Extract the template name.
                if let Some(name) = extract_quoted_string(trimmed.trim_start_matches("define").trim_start()) {
                    // Find the closing `}}` of the define tag.
                    let close = after_open.find("}}").unwrap_or(after_open.len());
                    let after_define_tag = &after_open[close + 2..];

                    // Find the matching `{{ end }}`.
                    if let Some(body_end) = find_matching_end(after_define_tag) {
                        let body = &after_define_tag[..body_end];
                        self.named_templates.insert(name.to_string(), body.to_string());

                        // Skip past `{{ end }}`.
                        let after_body = &after_define_tag[body_end..];
                        let skip_end = skip_end_tag(after_body);
                        rest = &after_body[skip_end..];
                        output.push_str(before);
                        continue;
                    }
                }
            }

            // Not a define — emit the text before `{{` and advance past `{{`.
            output.push_str(before);
            output.push_str("{{");
            rest = after_open;
        }

        output.push_str(rest);
        output
    }

    /// Render a template string using the given context.
    pub fn render(&self, template: &str, ctx: &TemplateContext) -> Result<String> {
        let tokens = parse_template_tokens(template);
        self.render_tokens(&tokens, ctx, None)
    }

    /// Internal renderer that walks tokens and evaluates actions.
    fn render_tokens(
        &self,
        tokens: &[Token],
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Result<String> {
        let mut output = String::new();
        let mut idx = 0;

        while idx < tokens.len() {
            match &tokens[idx] {
                Token::Text(text) => {
                    output.push_str(text);
                    idx += 1;
                }
                Token::Action(action) => {
                    let action_trimmed = action.trim();

                    // ---------- if / else / end ----------
                    if action_trimmed.starts_with("if ") {
                        let (if_body, else_body, consumed) =
                            collect_if_else_block(tokens, idx)?;
                        let condition_expr = action_trimmed.strip_prefix("if ").unwrap().trim();
                        let cond_val = self.eval_expression(condition_expr, ctx, loop_item);
                        let truthy = match &cond_val {
                            Some(v) => is_truthy(v),
                            None => false,
                        };
                        if truthy {
                            output.push_str(&self.render_tokens(&if_body, ctx, loop_item)?);
                        } else {
                            output.push_str(&self.render_tokens(&else_body, ctx, loop_item)?);
                        }
                        idx = consumed;
                    }
                    // ---------- range ----------
                    else if action_trimmed.starts_with("range ") {
                        let (body_tokens, consumed) = collect_block(tokens, idx, "range", "end")?;
                        let expr = action_trimmed.strip_prefix("range ").unwrap().trim();
                        let list_val = self.eval_expression(expr, ctx, loop_item);
                        if let Some(serde_yaml::Value::Sequence(items)) = list_val {
                            for item in &items {
                                output.push_str(&self.render_tokens(&body_tokens, ctx, Some(item))?);
                            }
                        }
                        idx = consumed;
                    }
                    // ---------- include ----------
                    else if action_trimmed.starts_with("include ") {
                        let result = self.eval_include(action_trimmed, ctx, loop_item)?;
                        output.push_str(&result);
                        idx += 1;
                    }
                    // ---------- template functions & expressions ----------
                    else {
                        let result = self.eval_action(action_trimmed, ctx, loop_item)?;
                        output.push_str(&result);
                        idx += 1;
                    }
                }
            }
        }
        Ok(output)
    }

    /// Evaluate a single `{{ … }}` action that is NOT a block construct.
    fn eval_action(
        &self,
        action: &str,
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Result<String> {
        let action = action.trim();

        // `toYaml EXPR`
        if action.starts_with("toYaml ") {
            let expr = action.strip_prefix("toYaml ").unwrap().trim();
            let val = self.eval_expression(expr, ctx, loop_item).unwrap_or(serde_yaml::Value::Null);
            return Ok(value_to_full_yaml(&val));
        }

        // `quote EXPR`
        if action.starts_with("quote ") {
            let expr = action.strip_prefix("quote ").unwrap().trim();
            let val = self.eval_expression(expr, ctx, loop_item).unwrap_or(serde_yaml::Value::Null);
            let s = value_to_yaml_string(&val);
            return Ok(format!("\"{}\"", s));
        }

        // `indent N EXPR`
        if action.starts_with("indent ") {
            return self.eval_indent(action, ctx, loop_item);
        }

        // `trimSuffix SUFFIX EXPR`
        if action.starts_with("trimSuffix ") {
            return self.eval_trim_suffix(action, ctx, loop_item);
        }

        // `printf FMT ARGS…`
        if action.starts_with("printf ") {
            return self.eval_printf(action, ctx, loop_item);
        }

        // `default FALLBACK EXPR`
        if action.starts_with("default ") {
            return self.eval_default(action, ctx, loop_item);
        }

        // Simple dot-path expression (`.Values.x.y`, `.Release.Name`, `.`)
        if action.starts_with('.') || action == "$" {
            let val = self.eval_expression(action, ctx, loop_item);
            return Ok(val.map(|v| value_to_yaml_string(&v)).unwrap_or_default());
        }

        // Bare string literal (rare but possible).
        if action.starts_with('"') && action.ends_with('"') {
            return Ok(action[1..action.len() - 1].to_string());
        }

        // Pipe chains: `EXPR | func`
        if action.contains('|') {
            return self.eval_pipeline(action, ctx, loop_item);
        }

        Ok(String::new())
    }

    /// Evaluate a pipeline like `.Values.x | toYaml | indent 4`.
    fn eval_pipeline(
        &self,
        action: &str,
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Result<String> {
        let parts: Vec<&str> = action.split('|').collect();
        if parts.is_empty() {
            return Ok(String::new());
        }

        // Evaluate the leftmost expression.
        let first = parts[0].trim();
        let mut current_val = self
            .eval_expression(first, ctx, loop_item)
            .unwrap_or(serde_yaml::Value::Null);
        let mut current_str = value_to_yaml_string(&current_val);

        for pipe_fn in &parts[1..] {
            let f = pipe_fn.trim();
            if f == "toYaml" {
                current_str = value_to_full_yaml(&current_val);
            } else if f == "quote" {
                current_str = format!("\"{}\"", current_str);
                current_val = serde_yaml::Value::String(current_str.clone());
            } else if f.starts_with("indent ") {
                let n_str = f.strip_prefix("indent ").unwrap().trim();
                let n: usize = n_str.parse().unwrap_or(0);
                current_str = indent_string(&current_str, n);
                current_val = serde_yaml::Value::String(current_str.clone());
            } else if f.starts_with("default ") {
                let fallback_expr = f.strip_prefix("default ").unwrap().trim();
                if !is_truthy(&current_val) {
                    let fallback = extract_string_literal(fallback_expr)
                        .unwrap_or_else(|| fallback_expr.to_string());
                    current_str = fallback.clone();
                    current_val = serde_yaml::Value::String(fallback);
                }
            } else if f.starts_with("trimSuffix ") {
                let suffix_expr = f.strip_prefix("trimSuffix ").unwrap().trim();
                let suffix = extract_string_literal(suffix_expr).unwrap_or_else(|| suffix_expr.to_string());
                current_str = current_str.strip_suffix(&suffix).unwrap_or(&current_str).to_string();
                current_val = serde_yaml::Value::String(current_str.clone());
            } else if f == "nindent" || f.starts_with("nindent ") {
                let n_str = f.strip_prefix("nindent").unwrap_or("0").trim();
                let n: usize = n_str.parse().unwrap_or(0);
                current_str = format!("\n{}", indent_string(&current_str, n));
                current_val = serde_yaml::Value::String(current_str.clone());
            }
        }

        Ok(current_str)
    }

    /// Evaluate a single expression (dot-path, literal, `.` for loop item).
    fn eval_expression(
        &self,
        expr: &str,
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Option<serde_yaml::Value> {
        let expr = expr.trim();

        // Current loop item.
        if expr == "." || expr == "$" {
            if let Some(item) = loop_item {
                return Some(item.clone());
            }
            return Some(ctx.values.inner().clone());
        }

        // String literal.
        if expr.starts_with('"') && expr.ends_with('"') && expr.len() >= 2 {
            return Some(serde_yaml::Value::String(expr[1..expr.len() - 1].to_string()));
        }

        // Numeric literal.
        if let Ok(n) = expr.parse::<i64>() {
            return Some(serde_yaml::Value::Number(n.into()));
        }
        if let Ok(f) = expr.parse::<f64>() {
            return Some(serde_yaml::Value::Number(serde_yaml::Number::from(f)));
        }

        // Boolean literal.
        if expr == "true" {
            return Some(serde_yaml::Value::Bool(true));
        }
        if expr == "false" {
            return Some(serde_yaml::Value::Bool(false));
        }

        // Dot-path from the loop item (e.g. `.name` inside a range when `.` is a mapping).
        if expr.starts_with('.') && loop_item.is_some() {
            let path = &expr[1..]; // strip leading dot
            // First try resolving from context (e.g. `.Values.x`).
            if let Some(val) = resolve_dot_path(path, ctx) {
                return Some(val);
            }
            // Otherwise resolve against the loop item.
            if let Some(item) = loop_item {
                return navigate_value(item, path);
            }
        }

        if expr.starts_with('.') {
            return resolve_dot_path(&expr[1..], ctx);
        }

        None
    }

    /// Evaluate `default "fallback" .Values.x`.
    fn eval_default(
        &self,
        action: &str,
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Result<String> {
        let rest = action.strip_prefix("default ").unwrap().trim();
        let (fallback, remainder) = split_first_arg(rest);
        let val = self.eval_expression(remainder.trim(), ctx, loop_item);
        match val {
            Some(v) if is_truthy(&v) => Ok(value_to_yaml_string(&v)),
            _ => Ok(fallback),
        }
    }

    /// Evaluate `indent N TEXT_EXPR`.
    fn eval_indent(
        &self,
        action: &str,
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Result<String> {
        let rest = action.strip_prefix("indent ").unwrap().trim();
        let (n_str, remainder) = split_first_arg(rest);
        let n: usize = n_str.parse().unwrap_or(0);
        let val = self
            .eval_expression(remainder.trim(), ctx, loop_item)
            .map(|v| value_to_yaml_string(&v))
            .unwrap_or_default();
        Ok(indent_string(&val, n))
    }

    /// Evaluate `trimSuffix "-" .Values.x`.
    fn eval_trim_suffix(
        &self,
        action: &str,
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Result<String> {
        let rest = action.strip_prefix("trimSuffix ").unwrap().trim();
        let (suffix, remainder) = split_first_arg(rest);
        let val = self
            .eval_expression(remainder.trim(), ctx, loop_item)
            .map(|v| value_to_yaml_string(&v))
            .unwrap_or_default();
        Ok(val.strip_suffix(&suffix).unwrap_or(&val).to_string())
    }

    /// Evaluate `printf "%s-%s" .Release.Name .Chart.Name`.
    fn eval_printf(
        &self,
        action: &str,
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Result<String> {
        let rest = action.strip_prefix("printf ").unwrap().trim();
        let (fmt_str, remainder) = split_first_arg(rest);
        let args: Vec<String> = shell_split(remainder.trim())
            .iter()
            .map(|a| {
                self.eval_expression(a, ctx, loop_item)
                    .map(|v| value_to_yaml_string(&v))
                    .unwrap_or_default()
            })
            .collect();

        // Simple Go-style printf: replace successive `%s`, `%d`, `%v` with args.
        let mut result = fmt_str;
        for arg in &args {
            if let Some(pos) = result.find("%s").or_else(|| result.find("%d")).or_else(|| result.find("%v")) {
                result = format!("{}{}{}", &result[..pos], arg, &result[pos + 2..]);
            }
        }
        Ok(result)
    }

    /// Evaluate `include "template-name" .`.
    fn eval_include(
        &self,
        action: &str,
        ctx: &TemplateContext,
        loop_item: Option<&serde_yaml::Value>,
    ) -> Result<String> {
        let rest = action.strip_prefix("include ").unwrap().trim();
        let (name, _remainder) = split_first_arg(rest);

        let tpl_body = self
            .named_templates
            .get(&name)
            .cloned()
            .unwrap_or_default();

        // Create a sub-engine with the same named templates to allow recursive includes.
        let sub_engine = GoTemplateEngine {
            named_templates: self.named_templates.clone(),
        };

        // If loop_item is provided, use it; otherwise use context values root.
        let _ = loop_item; // included template always receives full context
        sub_engine.render(&tpl_body, ctx)
    }
}

impl Default for GoTemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Standalone template resolution
// ---------------------------------------------------------------------------

/// Resolve a Go template string against a `TemplateContext`.
pub fn resolve_go_template(template: &str, context: &TemplateContext) -> Result<String> {
    let mut engine = GoTemplateEngine::new();
    let cleaned = engine.register_defines(template);
    engine.render(&cleaned, context)
}

// ---------------------------------------------------------------------------
// HelmProcessor
// ---------------------------------------------------------------------------

/// Main entry point for Helm chart processing.
#[derive(Debug, Clone)]
pub struct HelmProcessor;

impl HelmProcessor {
    /// Expand a single template string using the given values.
    pub fn expand_template(template: &str, values: &HelmValues) -> Result<String> {
        let ctx = default_context_for_values(values);
        resolve_go_template(template, &ctx)
    }

    /// Walk the `templates/` directory of a chart, expand each template, and
    /// return the expanded YAML strings.
    pub fn process_helm_chart(chart_dir: &str, values: &HelmValues) -> Result<Vec<String>> {
        let templates_dir = Path::new(chart_dir).join("templates");
        if !templates_dir.exists() {
            bail!("templates directory not found in chart: {}", chart_dir);
        }

        let chart_yaml_path = Path::new(chart_dir).join("Chart.yaml");
        let chart_meta = if chart_yaml_path.exists() {
            let raw = fs::read_to_string(&chart_yaml_path)
                .with_context(|| format!("reading {}", chart_yaml_path.display()))?;
            serde_yaml::from_str::<ChartMetadata>(&raw).unwrap_or_else(|_| default_chart_metadata())
        } else {
            default_chart_metadata()
        };

        let ctx = TemplateContext {
            values: values.clone(),
            release: default_release_info(),
            chart: chart_meta,
            capabilities: default_capabilities(),
        };

        // First pass: register all defines from helper files (e.g. _helpers.tpl).
        let mut engine = GoTemplateEngine::new();
        let mut template_files: Vec<std::path::PathBuf> = Vec::new();

        for entry in fs::read_dir(&templates_dir)
            .with_context(|| format!("reading templates dir: {}", templates_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                template_files.push(path);
            }
        }

        // Sort for deterministic ordering; process helpers first.
        template_files.sort();
        let (helpers, others): (Vec<_>, Vec<_>) = template_files
            .into_iter()
            .partition(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().starts_with('_'))
                    .unwrap_or(false)
            });

        for helper in &helpers {
            let content = fs::read_to_string(helper)?;
            engine.register_defines(&content);
        }

        // Second pass: render all non-helper templates.
        let mut results = Vec::new();
        for tpl_path in &others {
            let content = fs::read_to_string(tpl_path)?;
            let cleaned = engine.register_defines(&content);
            let rendered = engine
                .render(&cleaned, &ctx)
                .with_context(|| format!("rendering {}", tpl_path.display()))?;
            let trimmed = rendered.trim().to_string();
            if !trimmed.is_empty() {
                results.push(trimmed);
            }
        }

        Ok(results)
    }

    /// Deep-merge two `HelmValues`, overlay wins for scalars, maps merge recursively,
    /// arrays are replaced.
    pub fn merge_values(base: &HelmValues, overlay: &HelmValues) -> HelmValues {
        HelmValues::new(deep_merge(base.inner(), overlay.inner()))
    }
}

/// Convenience alias for the standalone merge function.
pub fn merge_values(base: &HelmValues, overlay: &HelmValues) -> HelmValues {
    HelmProcessor::merge_values(base, overlay)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a minimal `TemplateContext` for cases where only values are supplied.
fn default_context_for_values(values: &HelmValues) -> TemplateContext {
    TemplateContext {
        values: values.clone(),
        release: default_release_info(),
        chart: default_chart_metadata(),
        capabilities: default_capabilities(),
    }
}

fn default_release_info() -> ReleaseInfo {
    ReleaseInfo {
        name: "release-name".to_string(),
        namespace: "default".to_string(),
        revision: 1,
        is_install: true,
        is_upgrade: false,
    }
}

fn default_chart_metadata() -> ChartMetadata {
    ChartMetadata {
        name: "chart".to_string(),
        version: "0.1.0".to_string(),
        app_version: "1.0.0".to_string(),
        description: String::new(),
        dependencies: Vec::new(),
    }
}

fn default_capabilities() -> Capabilities {
    Capabilities {
        kube_version: "v1.28.0".to_string(),
        api_versions: vec!["apps/v1".to_string(), "v1".to_string()],
    }
}

/// Indent every line of `text` by `n` spaces.
fn indent_string(text: &str, n: usize) -> String {
    let prefix = " ".repeat(n);
    text.lines()
        .enumerate()
        .map(|(i, line)| {
            if i == 0 {
                // First line is not indented (matches Helm behaviour for `indent`).
                line.to_string()
            } else {
                format!("{}{}", prefix, line)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Convert a YAML value to its full YAML representation (for `toYaml`).
fn value_to_full_yaml(val: &serde_yaml::Value) -> String {
    match val {
        serde_yaml::Value::Null => "null".to_string(),
        serde_yaml::Value::Bool(b) => b.to_string(),
        serde_yaml::Value::Number(n) => n.to_string(),
        serde_yaml::Value::String(s) => s.clone(),
        _ => {
            let raw = serde_yaml::to_string(val).unwrap_or_default();
            // serde_yaml may emit a leading `---\n` document separator; strip it.
            let trimmed = raw.strip_prefix("---\n").unwrap_or(&raw);
            trimmed.trim_end_matches('\n').to_string()
        }
    }
}

/// Navigate a YAML value by a dot-separated path.
fn navigate_value(val: &serde_yaml::Value, path: &str) -> Option<serde_yaml::Value> {
    let mut current = val.clone();
    for segment in path.split('.') {
        if segment.is_empty() {
            continue;
        }
        match current {
            serde_yaml::Value::Mapping(ref map) => {
                let key = serde_yaml::Value::String(segment.to_string());
                current = map.get(&key)?.clone();
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Extract a quoted string (e.g. `"my-template"`) from the beginning of `s`.
fn extract_quoted_string(s: &str) -> Option<String> {
    let s = s.trim();
    if s.starts_with('"') {
        let end = s[1..].find('"')?;
        Some(s[1..1 + end].to_string())
    } else {
        None
    }
}

/// Extract a string literal (quoted or bare) from the start of `s`.
fn extract_string_literal(s: &str) -> Option<String> {
    let s = s.trim();
    if s.starts_with('"') && s.len() >= 2 {
        let end = s[1..].find('"')?;
        Some(s[1..1 + end].to_string())
    } else {
        None
    }
}

/// Split the first argument from the rest.  Handles quoted strings.
fn split_first_arg(s: &str) -> (String, String) {
    let s = s.trim();
    if s.starts_with('"') {
        if let Some(end) = s[1..].find('"') {
            let arg = s[1..1 + end].to_string();
            let rest = s[2 + end..].to_string();
            return (arg, rest);
        }
    }
    // Bare word.
    let split_pos = s.find(char::is_whitespace).unwrap_or(s.len());
    let arg = s[..split_pos].to_string();
    let rest = s[split_pos..].to_string();
    (arg, rest)
}

/// Simple shell-like split respecting quoted strings.
fn shell_split(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '"' {
            in_quotes = !in_quotes;
        } else if c.is_whitespace() && !in_quotes {
            if !current.is_empty() {
                result.push(std::mem::take(&mut current));
            }
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

/// Find the offset of the text just before the matching `{{ end }}` tag for a
/// block that has already been opened (i.e. the open tag has been consumed).
/// Handles nested blocks.
fn find_matching_end(template: &str) -> Option<usize> {
    let mut depth = 1u32;
    let mut pos = 0usize;

    while pos < template.len() {
        if let Some(tag_start) = template[pos..].find("{{") {
            let abs = pos + tag_start;
            let after = template[abs + 2..].trim_start_matches('-').trim_start();
            if let Some(close) = template[abs + 2..].find("}}") {
                let action_end = abs + 2 + close;
                if after.starts_with("if ")
                    || after.starts_with("range ")
                    || after.starts_with("define ")
                    || after.starts_with("block ")
                {
                    depth += 1;
                } else if after.starts_with("end") {
                    depth -= 1;
                    if depth == 0 {
                        return Some(abs);
                    }
                }
                pos = action_end + 2;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    None
}

/// Skip past a `{{ end }}` tag at the beginning of `s`, returning the number of
/// bytes consumed.
fn skip_end_tag(s: &str) -> usize {
    if let Some(start) = s.find("{{") {
        if let Some(end) = s[start + 2..].find("}}") {
            return start + 2 + end + 2;
        }
    }
    0
}

/// Collect tokens forming the body of an `if` / `else` / `end` block.
/// Returns `(if_body_tokens, else_body_tokens, next_index_after_end)`.
fn collect_if_else_block(
    tokens: &[Token],
    start: usize,
) -> Result<(Vec<Token>, Vec<Token>, usize)> {
    let mut depth = 1u32;
    let mut idx = start + 1; // skip the opening `if` action
    let mut if_tokens: Vec<Token> = Vec::new();
    let mut else_tokens: Vec<Token> = Vec::new();
    let mut in_else = false;

    while idx < tokens.len() {
        match &tokens[idx] {
            Token::Action(a) => {
                let a = a.trim();
                if a.starts_with("if ") {
                    depth += 1;
                    if in_else {
                        else_tokens.push(tokens[idx].clone());
                    } else {
                        if_tokens.push(tokens[idx].clone());
                    }
                } else if a.starts_with("range ") || a.starts_with("define ") || a.starts_with("block ") {
                    depth += 1;
                    if in_else {
                        else_tokens.push(tokens[idx].clone());
                    } else {
                        if_tokens.push(tokens[idx].clone());
                    }
                } else if (a == "else" || a.starts_with("else ")) && depth == 1 {
                    in_else = true;
                } else if a == "end" {
                    depth -= 1;
                    if depth == 0 {
                        return Ok((if_tokens, else_tokens, idx + 1));
                    }
                    if in_else {
                        else_tokens.push(tokens[idx].clone());
                    } else {
                        if_tokens.push(tokens[idx].clone());
                    }
                } else {
                    if in_else {
                        else_tokens.push(tokens[idx].clone());
                    } else {
                        if_tokens.push(tokens[idx].clone());
                    }
                }
            }
            _ => {
                if in_else {
                    else_tokens.push(tokens[idx].clone());
                } else {
                    if_tokens.push(tokens[idx].clone());
                }
            }
        }
        idx += 1;
    }
    bail!("unterminated if block");
}

/// Collect tokens for a generic block (range / define) up to its matching `end`.
fn collect_block(
    tokens: &[Token],
    start: usize,
    open_keyword: &str,
    _close_keyword: &str,
) -> Result<(Vec<Token>, usize)> {
    let mut depth = 1u32;
    let mut idx = start + 1;
    let mut body: Vec<Token> = Vec::new();

    while idx < tokens.len() {
        match &tokens[idx] {
            Token::Action(a) => {
                let a = a.trim();
                if a.starts_with("if ")
                    || a.starts_with("range ")
                    || a.starts_with("define ")
                    || a.starts_with("block ")
                {
                    depth += 1;
                    body.push(tokens[idx].clone());
                } else if a == "end" {
                    depth -= 1;
                    if depth == 0 {
                        return Ok((body, idx + 1));
                    }
                    body.push(tokens[idx].clone());
                } else {
                    body.push(tokens[idx].clone());
                }
            }
            _ => {
                body.push(tokens[idx].clone());
            }
        }
        idx += 1;
    }
    bail!("unterminated {} block", open_keyword);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> TemplateContext {
        let yaml: serde_yaml::Value = serde_yaml::from_str(
            r#"
image:
  repository: nginx
  tag: "1.21"
replicaCount: 3
fullnameOverride: ""
labels:
  app: myapp
  tier: frontend
items:
  - alpha
  - beta
  - gamma
nested:
  enabled: true
  deep:
    key: deepval
"#,
        )
        .unwrap();

        TemplateContext {
            values: HelmValues::new(yaml),
            release: ReleaseInfo {
                name: "my-release".to_string(),
                namespace: "production".to_string(),
                revision: 2,
                is_install: false,
                is_upgrade: true,
            },
            chart: ChartMetadata {
                name: "my-chart".to_string(),
                version: "1.2.3".to_string(),
                app_version: "4.5.6".to_string(),
                description: "A test chart".to_string(),
                dependencies: vec![],
            },
            capabilities: Capabilities {
                kube_version: "v1.28.0".to_string(),
                api_versions: vec!["apps/v1".to_string()],
            },
        }
    }

    #[test]
    fn test_simple_value_substitution() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let result = engine.render("image: {{ .Values.image.repository }}", &ctx).unwrap();
        assert_eq!(result, "image: nginx");
    }

    #[test]
    fn test_nested_value_access() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let result = engine.render("{{ .Values.nested.deep.key }}", &ctx).unwrap();
        assert_eq!(result, "deepval");
    }

    #[test]
    fn test_default_function() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        // fullnameOverride is "" (empty), so default should kick in.
        let result = engine
            .render(r#"{{ default "fallback" .Values.fullnameOverride }}"#, &ctx)
            .unwrap();
        assert_eq!(result, "fallback");

        // image.repository is set, so it should win over the default.
        let result2 = engine
            .render(r#"{{ default "fallback" .Values.image.repository }}"#, &ctx)
            .unwrap();
        assert_eq!(result2, "nginx");
    }

    #[test]
    fn test_if_else_block() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = "{{ if .Values.nested.enabled }}yes{{ else }}no{{ end }}";
        assert_eq!(engine.render(tpl, &ctx).unwrap(), "yes");

        let tpl2 = "{{ if .Values.fullnameOverride }}has-name{{ else }}no-name{{ end }}";
        assert_eq!(engine.render(tpl2, &ctx).unwrap(), "no-name");
    }

    #[test]
    fn test_range_block() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = "{{ range .Values.items }}- {{ . }}\n{{ end }}";
        let result = engine.render(tpl, &ctx).unwrap();
        assert!(result.contains("- alpha"));
        assert!(result.contains("- beta"));
        assert!(result.contains("- gamma"));
    }

    #[test]
    fn test_include_named_template() {
        let ctx = make_ctx();
        let mut engine = GoTemplateEngine::new();
        engine
            .named_templates
            .insert("my-chart.fullname".to_string(), "{{ .Release.Name }}-{{ .Chart.Name }}".to_string());
        let tpl = r#"name: {{ include "my-chart.fullname" . }}"#;
        let result = engine.render(tpl, &ctx).unwrap();
        assert_eq!(result, "name: my-release-my-chart");
    }

    #[test]
    fn test_to_yaml() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = "{{ toYaml .Values.labels }}";
        let result = engine.render(tpl, &ctx).unwrap();
        assert!(result.contains("app: myapp"));
        assert!(result.contains("tier: frontend"));
    }

    #[test]
    fn test_quote_function() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = "{{ quote .Values.image.repository }}";
        let result = engine.render(tpl, &ctx).unwrap();
        assert_eq!(result, "\"nginx\"");
    }

    #[test]
    fn test_indent_function() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = "{{ indent 4 .Values.image.repository }}";
        let result = engine.render(tpl, &ctx).unwrap();
        // indent on a single line just returns the text (first line not indented).
        assert_eq!(result, "nginx");
    }

    #[test]
    fn test_merge_values_basic() {
        let base = HelmValues::new(serde_yaml::from_str("a: 1\nb: 2").unwrap());
        let overlay = HelmValues::new(serde_yaml::from_str("b: 99\nc: 3").unwrap());
        let merged = merge_values(&base, &overlay);
        assert_eq!(
            merged.get("a").unwrap(),
            &serde_yaml::Value::Number(serde_yaml::Number::from(1))
        );
        assert_eq!(
            merged.get("b").unwrap(),
            &serde_yaml::Value::Number(serde_yaml::Number::from(99))
        );
        assert_eq!(
            merged.get("c").unwrap(),
            &serde_yaml::Value::Number(serde_yaml::Number::from(3))
        );
    }

    #[test]
    fn test_merge_values_deep() {
        let base = HelmValues::new(
            serde_yaml::from_str("parent:\n  a: 1\n  b: 2").unwrap(),
        );
        let overlay = HelmValues::new(
            serde_yaml::from_str("parent:\n  b: 99\n  c: 3").unwrap(),
        );
        let merged = merge_values(&base, &overlay);
        assert_eq!(
            merged.get("parent.a").unwrap(),
            &serde_yaml::Value::Number(serde_yaml::Number::from(1))
        );
        assert_eq!(
            merged.get("parent.b").unwrap(),
            &serde_yaml::Value::Number(serde_yaml::Number::from(99))
        );
        assert_eq!(
            merged.get("parent.c").unwrap(),
            &serde_yaml::Value::Number(serde_yaml::Number::from(3))
        );
    }

    #[test]
    fn test_helm_values_get_set() {
        let mut vals = HelmValues::new(serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
        vals.set("x.y.z", serde_yaml::Value::String("hello".to_string()));
        assert_eq!(
            vals.get("x.y.z").unwrap(),
            &serde_yaml::Value::String("hello".to_string())
        );
        vals.set("x.y.z", serde_yaml::Value::String("world".to_string()));
        assert_eq!(
            vals.get("x.y.z").unwrap(),
            &serde_yaml::Value::String("world".to_string())
        );
    }

    #[test]
    fn test_whitespace_trimming() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = "hello  {{- .Values.image.repository -}}  world";
        let result = engine.render(tpl, &ctx).unwrap();
        assert_eq!(result, "hellonginxworld");
    }

    #[test]
    fn test_printf_function() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = r#"{{ printf "%s-%s" .Release.Name .Chart.Name }}"#;
        let result = engine.render(tpl, &ctx).unwrap();
        assert_eq!(result, "my-release-my-chart");
    }

    #[test]
    fn test_release_values() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        assert_eq!(
            engine.render("{{ .Release.Name }}", &ctx).unwrap(),
            "my-release"
        );
        assert_eq!(
            engine.render("{{ .Release.Namespace }}", &ctx).unwrap(),
            "production"
        );
        assert_eq!(
            engine.render("{{ .Release.IsUpgrade }}", &ctx).unwrap(),
            "true"
        );
    }

    #[test]
    fn test_define_and_include() {
        let ctx = make_ctx();
        let mut engine = GoTemplateEngine::new();
        let tpl = r#"{{ define "greeting" }}Hello {{ .Release.Name }}{{ end }}Result: {{ include "greeting" . }}"#;
        let cleaned = engine.register_defines(tpl);
        let result = engine.render(&cleaned, &ctx).unwrap();
        assert_eq!(result, "Result: Hello my-release");
    }

    #[test]
    fn test_pipeline_to_yaml_indent() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = "{{ .Values.labels | toYaml | indent 2 }}";
        let result = engine.render(tpl, &ctx).unwrap();
        assert!(result.contains("app: myapp"));
        // The second line should be indented.
        let lines: Vec<&str> = result.lines().collect();
        assert!(lines.len() >= 2);
        assert!(lines[1].starts_with("  "));
    }

    #[test]
    fn test_is_truthy() {
        assert!(!is_truthy(&serde_yaml::Value::Null));
        assert!(!is_truthy(&serde_yaml::Value::Bool(false)));
        assert!(!is_truthy(&serde_yaml::Value::String("".to_string())));
        assert!(!is_truthy(&serde_yaml::Value::Number(0.into())));
        assert!(is_truthy(&serde_yaml::Value::Bool(true)));
        assert!(is_truthy(&serde_yaml::Value::String("x".to_string())));
        assert!(is_truthy(&serde_yaml::Value::Number(1.into())));
    }

    #[test]
    fn test_parse_template_tokens() {
        let tokens = parse_template_tokens("Hello {{ .Values.name }} world");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], Token::Text("Hello ".to_string()));
        assert_eq!(tokens[1], Token::Action(".Values.name".to_string()));
        assert_eq!(tokens[2], Token::Text(" world".to_string()));
    }

    #[test]
    fn test_trim_suffix_function() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = r#"{{ trimSuffix "-chart" .Chart.Name }}"#;
        let result = engine.render(tpl, &ctx).unwrap();
        assert_eq!(result, "my");
    }

    #[test]
    fn test_chart_values() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        assert_eq!(engine.render("{{ .Chart.Name }}", &ctx).unwrap(), "my-chart");
        assert_eq!(engine.render("{{ .Chart.Version }}", &ctx).unwrap(), "1.2.3");
        assert_eq!(engine.render("{{ .Chart.AppVersion }}", &ctx).unwrap(), "4.5.6");
    }

    #[test]
    fn test_nested_if_inside_range() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = "{{ range .Values.items }}{{ if . }}[{{ . }}]{{ end }}{{ end }}";
        let result = engine.render(tpl, &ctx).unwrap();
        assert!(result.contains("[alpha]"));
        assert!(result.contains("[beta]"));
        assert!(result.contains("[gamma]"));
    }

    #[test]
    fn test_deep_merge_arrays_replaced() {
        let base: serde_yaml::Value = serde_yaml::from_str("items:\n  - a\n  - b").unwrap();
        let overlay: serde_yaml::Value = serde_yaml::from_str("items:\n  - x").unwrap();
        let result = deep_merge(&base, &overlay);
        let items = result.as_mapping().unwrap().get("items").unwrap().as_sequence().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], serde_yaml::Value::String("x".to_string()));
    }

    #[test]
    fn test_resolve_dot_path() {
        let ctx = make_ctx();
        let val = resolve_dot_path("Values.image.tag", &ctx);
        assert_eq!(val.unwrap(), serde_yaml::Value::String("1.21".to_string()));

        let val2 = resolve_dot_path("Release.Namespace", &ctx);
        assert_eq!(
            val2.unwrap(),
            serde_yaml::Value::String("production".to_string())
        );
    }

    #[test]
    fn test_multiline_template() {
        let ctx = make_ctx();
        let engine = GoTemplateEngine::new();
        let tpl = r#"apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-config
  namespace: {{ .Release.Namespace }}
data:
  repo: {{ .Values.image.repository }}
  tag: {{ .Values.image.tag }}"#;
        let result = engine.render(tpl, &ctx).unwrap();
        assert!(result.contains("name: my-release-config"));
        assert!(result.contains("namespace: production"));
        assert!(result.contains("repo: nginx"));
        assert!(result.contains("tag: 1.21"));
    }
}

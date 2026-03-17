//! Sonification templates for SoniType.
//!
//! Parameterised DSL snippets with variable substitution, conditionals,
//! and loop expansion for common sonification patterns.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TemplateParameter
// ---------------------------------------------------------------------------

/// Type of a template parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamType {
    Float,
    Int,
    String,
    Bool,
}

/// Metadata for a template parameter.
#[derive(Debug, Clone)]
pub struct TemplateParameter {
    pub name: String,
    pub param_type: ParamType,
    pub default_value: String,
    pub description: String,
}

impl TemplateParameter {
    pub fn new(
        name: impl Into<String>,
        param_type: ParamType,
        default_value: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            param_type,
            default_value: default_value.into(),
            description: description.into(),
        }
    }

    /// Validate a value string against this parameter's type.
    pub fn validate_value(&self, value: &str) -> bool {
        match self.param_type {
            ParamType::Float => value.parse::<f64>().is_ok(),
            ParamType::Int => value.parse::<i64>().is_ok(),
            ParamType::String => true,
            ParamType::Bool => value == "true" || value == "false",
        }
    }
}

// ---------------------------------------------------------------------------
// TemplateEngine
// ---------------------------------------------------------------------------

/// Template expansion engine with variable substitution, conditionals, and loops.
#[derive(Debug, Clone)]
pub struct TemplateEngine {
    pub variables: HashMap<String, String>,
}

impl TemplateEngine {
    pub fn new() -> Self {
        Self { variables: HashMap::new() }
    }

    /// Set a variable value.
    pub fn set(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.variables.insert(name.into(), value.into());
    }

    /// Set multiple variables from a map.
    pub fn set_all(&mut self, vars: HashMap<String, String>) {
        self.variables.extend(vars);
    }

    /// Expand a template string, replacing `{{var}}` with variable values.
    pub fn expand(&self, template: &str) -> String {
        let mut result = template.to_string();

        // Process conditional blocks: {{#if VAR}}...{{/if}}
        result = self.expand_conditionals(&result);

        // Process loop blocks: {{#each VAR}}...{{/each}}
        result = self.expand_loops(&result);

        // Variable substitution: {{VAR}}
        for (key, value) in &self.variables {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        result
    }

    /// Expand conditional blocks `{{#if VAR}}...{{/if}}`.
    fn expand_conditionals(&self, input: &str) -> String {
        let mut result = input.to_string();
        loop {
            let start_tag = "{{#if ";
            let start_pos = match result.find(start_tag) {
                Some(p) => p,
                None => break,
            };
            let tag_end = match result[start_pos + start_tag.len()..].find("}}") {
                Some(p) => start_pos + start_tag.len() + p,
                None => break,
            };
            let var_name = result[start_pos + start_tag.len()..tag_end].trim().to_string();
            let block_start = tag_end + 2;

            // Look for optional {{#else}} and {{/if}}
            let end_tag = "{{/if}}";
            let end_pos = match result[block_start..].find(end_tag) {
                Some(p) => block_start + p,
                None => break,
            };

            let block_content = &result[block_start..end_pos];

            // Check for else branch
            let (true_block, false_block) = if let Some(else_pos) = block_content.find("{{#else}}") {
                (&block_content[..else_pos], &block_content[else_pos + 9..])
            } else {
                (block_content, "")
            };

            let condition_met = self.variables.get(&var_name)
                .map(|v| v != "false" && v != "0" && !v.is_empty())
                .unwrap_or(false);

            let replacement = if condition_met {
                true_block.to_string()
            } else {
                false_block.to_string()
            };

            result = format!(
                "{}{}{}",
                &result[..start_pos],
                replacement,
                &result[end_pos + end_tag.len()..],
            );
        }
        result
    }

    /// Expand loop blocks `{{#each ITEMS}}...{{/each}}`.
    /// ITEMS variable should contain comma-separated values.
    fn expand_loops(&self, input: &str) -> String {
        let mut result = input.to_string();
        loop {
            let start_tag = "{{#each ";
            let start_pos = match result.find(start_tag) {
                Some(p) => p,
                None => break,
            };
            let tag_end = match result[start_pos + start_tag.len()..].find("}}") {
                Some(p) => start_pos + start_tag.len() + p,
                None => break,
            };
            let var_name = result[start_pos + start_tag.len()..tag_end].trim().to_string();
            let block_start = tag_end + 2;

            let end_tag = "{{/each}}";
            let end_pos = match result[block_start..].find(end_tag) {
                Some(p) => block_start + p,
                None => break,
            };

            let block_template = &result[block_start..end_pos];

            let items: Vec<&str> = self.variables.get(&var_name)
                .map(|v| v.split(',').map(|s| s.trim()).collect())
                .unwrap_or_default();

            let mut expanded = String::new();
            for (i, item) in items.iter().enumerate() {
                let mut iteration = block_template.to_string();
                iteration = iteration.replace("{{item}}", item);
                iteration = iteration.replace("{{index}}", &i.to_string());
                expanded.push_str(&iteration);
            }

            result = format!(
                "{}{}{}",
                &result[..start_pos],
                expanded,
                &result[end_pos + end_tag.len()..],
            );
        }
        result
    }

    /// Validate that all required parameters are set.
    pub fn validate_params(&self, params: &[TemplateParameter]) -> Vec<String> {
        let mut errors = Vec::new();
        for p in params {
            if let Some(val) = self.variables.get(&p.name) {
                if !p.validate_value(val) {
                    errors.push(format!(
                        "Parameter '{}': value '{}' does not match type {:?}",
                        p.name, val, p.param_type
                    ));
                }
            }
            // If not set, the default will be used (no error).
        }
        errors
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in template strings
// ---------------------------------------------------------------------------

/// Single-stream time series sonification template.
pub const TEMPLATE_TIME_SERIES_SINGLE: &str = r#"
stream time_series {
    data source = "{{data_source}}";
    mapping {
        {{value_field}} -> pitch({{pitch_min}}Hz, {{pitch_max}}Hz);
        {{#if loudness_field}}
        {{loudness_field}} -> loudness({{loudness_min}}dB, {{loudness_max}}dB);
        {{/if}}
    }
    tempo = {{bpm}} bpm;
    timbre = "{{timbre}}";
}
"#;

/// Multi-stream parallel sonification template.
pub const TEMPLATE_MULTI_STREAM: &str = r#"
compose parallel_streams {
    {{#each streams}}
    stream stream_{{index}} {
        data source = "{{data_source}}";
        field = "{{item}}";
        mapping {
            {{item}} -> pitch({{pitch_min}}Hz, {{pitch_max}}Hz);
        }
        timbre = palette[{{index}}];
    }
    {{/each}}
    tempo = {{bpm}} bpm;
}
"#;

/// Sequential scanning template (sweep through bins/categories).
pub const TEMPLATE_SEQUENTIAL_SCAN: &str = r#"
compose sequential_scan {
    {{#each bins}}
    event bin_{{index}} at {{index}} * {{step_duration}}s {
        pitch = {{item}}Hz;
        loudness = {{loudness}}dB;
        duration = {{step_duration}}s;
    }
    {{/each}}
}
"#;

/// Alert / notification template.
pub const TEMPLATE_ALERT: &str = r#"
alert {{alert_name}} {
    urgency = {{urgency}};
    pitch = {{pitch}}Hz;
    repetitions = {{repetitions}};
    interval = {{interval}}s;
    {{#if escalate}}
    escalation = true;
    {{/if}}
}
"#;

// ---------------------------------------------------------------------------
// TemplateLibrary
// ---------------------------------------------------------------------------

/// A named template with its parameter specifications.
#[derive(Debug, Clone)]
pub struct TemplateDefinition {
    pub name: String,
    pub description: String,
    pub template: String,
    pub parameters: Vec<TemplateParameter>,
}

/// Collection of named templates.
#[derive(Debug, Clone)]
pub struct TemplateLibrary {
    pub templates: HashMap<String, TemplateDefinition>,
}

impl TemplateLibrary {
    pub fn new() -> Self {
        Self { templates: HashMap::new() }
    }

    /// Create a library with all built-in templates.
    pub fn with_builtins() -> Self {
        let mut lib = Self::new();

        lib.register(TemplateDefinition {
            name: "time_series_single".into(),
            description: "Single-stream time series sonification".into(),
            template: TEMPLATE_TIME_SERIES_SINGLE.into(),
            parameters: vec![
                TemplateParameter::new("data_source", ParamType::String, "input.csv", "Path to data source"),
                TemplateParameter::new("value_field", ParamType::String, "value", "Data field to sonify"),
                TemplateParameter::new("pitch_min", ParamType::Float, "200", "Minimum pitch in Hz"),
                TemplateParameter::new("pitch_max", ParamType::Float, "2000", "Maximum pitch in Hz"),
                TemplateParameter::new("bpm", ParamType::Float, "120", "Tempo in beats per minute"),
                TemplateParameter::new("timbre", ParamType::String, "flute", "Timbre preset name"),
                TemplateParameter::new("loudness_field", ParamType::String, "", "Optional loudness field"),
                TemplateParameter::new("loudness_min", ParamType::Float, "-40", "Min loudness dB"),
                TemplateParameter::new("loudness_max", ParamType::Float, "-6", "Max loudness dB"),
            ],
        });

        lib.register(TemplateDefinition {
            name: "multi_stream".into(),
            description: "Multi-stream parallel sonification".into(),
            template: TEMPLATE_MULTI_STREAM.into(),
            parameters: vec![
                TemplateParameter::new("data_source", ParamType::String, "input.csv", "Path to data source"),
                TemplateParameter::new("streams", ParamType::String, "temp,pressure,humidity", "Comma-separated field names"),
                TemplateParameter::new("pitch_min", ParamType::Float, "200", "Minimum pitch in Hz"),
                TemplateParameter::new("pitch_max", ParamType::Float, "2000", "Maximum pitch in Hz"),
                TemplateParameter::new("bpm", ParamType::Float, "120", "Tempo in BPM"),
            ],
        });

        lib.register(TemplateDefinition {
            name: "sequential_scan".into(),
            description: "Sequential scanning through bins".into(),
            template: TEMPLATE_SEQUENTIAL_SCAN.into(),
            parameters: vec![
                TemplateParameter::new("bins", ParamType::String, "200,400,600,800", "Comma-separated pitch values"),
                TemplateParameter::new("loudness", ParamType::Float, "-12", "Loudness in dB"),
                TemplateParameter::new("step_duration", ParamType::Float, "0.5", "Duration per step in seconds"),
            ],
        });

        lib.register(TemplateDefinition {
            name: "alert".into(),
            description: "Alert / notification sound".into(),
            template: TEMPLATE_ALERT.into(),
            parameters: vec![
                TemplateParameter::new("alert_name", ParamType::String, "warning", "Alert name"),
                TemplateParameter::new("urgency", ParamType::String, "medium", "Urgency level"),
                TemplateParameter::new("pitch", ParamType::Float, "880", "Pitch in Hz"),
                TemplateParameter::new("repetitions", ParamType::Int, "3", "Number of repetitions"),
                TemplateParameter::new("interval", ParamType::Float, "0.3", "Inter-onset interval"),
                TemplateParameter::new("escalate", ParamType::Bool, "false", "Enable pitch escalation"),
            ],
        });

        lib
    }

    pub fn register(&mut self, definition: TemplateDefinition) {
        self.templates.insert(definition.name.clone(), definition);
    }

    pub fn get(&self, name: &str) -> Option<&TemplateDefinition> {
        self.templates.get(name)
    }

    pub fn list(&self) -> Vec<&TemplateDefinition> {
        self.templates.values().collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.templates.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.templates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }

    /// Expand a named template with the given variable assignments.
    pub fn expand(&self, name: &str, vars: HashMap<String, String>) -> Option<String> {
        let def = self.get(name)?;
        let mut engine = TemplateEngine::new();

        // Apply defaults first.
        for param in &def.parameters {
            if !param.default_value.is_empty() {
                engine.set(&param.name, &param.default_value);
            }
        }
        // Override with provided values.
        engine.set_all(vars);
        Some(engine.expand(&def.template))
    }

    /// Validate variables against a template's parameter specs.
    pub fn validate(&self, name: &str, vars: &HashMap<String, String>) -> Vec<String> {
        let Some(def) = self.get(name) else {
            return vec![format!("Template '{}' not found", name)];
        };
        let mut engine = TemplateEngine::new();
        for (k, v) in vars {
            engine.set(k, v);
        }
        engine.validate_params(&def.parameters)
    }
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_substitution() {
        let mut engine = TemplateEngine::new();
        engine.set("name", "Alice");
        let result = engine.expand("Hello, {{name}}!");
        assert_eq!(result, "Hello, Alice!");
    }

    #[test]
    fn test_multiple_variables() {
        let mut engine = TemplateEngine::new();
        engine.set("x", "10");
        engine.set("y", "20");
        let result = engine.expand("{{x}} + {{y}}");
        assert_eq!(result, "10 + 20");
    }

    #[test]
    fn test_conditional_true() {
        let mut engine = TemplateEngine::new();
        engine.set("show", "true");
        let result = engine.expand("{{#if show}}visible{{/if}}");
        assert_eq!(result, "visible");
    }

    #[test]
    fn test_conditional_false() {
        let mut engine = TemplateEngine::new();
        engine.set("show", "false");
        let result = engine.expand("{{#if show}}visible{{#else}}hidden{{/if}}");
        assert_eq!(result, "hidden");
    }

    #[test]
    fn test_conditional_missing_var() {
        let engine = TemplateEngine::new();
        let result = engine.expand("{{#if missing}}yes{{#else}}no{{/if}}");
        assert_eq!(result, "no");
    }

    #[test]
    fn test_each_loop() {
        let mut engine = TemplateEngine::new();
        engine.set("items", "a,b,c");
        let result = engine.expand("{{#each items}}[{{item}}]{{/each}}");
        assert_eq!(result, "[a][b][c]");
    }

    #[test]
    fn test_each_loop_with_index() {
        let mut engine = TemplateEngine::new();
        engine.set("items", "x,y");
        let result = engine.expand("{{#each items}}{{index}}:{{item}} {{/each}}");
        assert_eq!(result, "0:x 1:y ");
    }

    #[test]
    fn test_template_parameter_validation() {
        let p = TemplateParameter::new("freq", ParamType::Float, "440", "Frequency");
        assert!(p.validate_value("440.0"));
        assert!(p.validate_value("200"));
        assert!(!p.validate_value("abc"));
    }

    #[test]
    fn test_template_library_builtins() {
        let lib = TemplateLibrary::with_builtins();
        assert!(lib.contains("time_series_single"));
        assert!(lib.contains("multi_stream"));
        assert!(lib.contains("alert"));
        assert!(lib.len() >= 4);
    }

    #[test]
    fn test_template_expand_time_series() {
        let lib = TemplateLibrary::with_builtins();
        let mut vars = HashMap::new();
        vars.insert("data_source".into(), "data.csv".into());
        vars.insert("value_field".into(), "temperature".into());
        vars.insert("pitch_min".into(), "300".into());
        vars.insert("pitch_max".into(), "3000".into());
        vars.insert("bpm".into(), "100".into());
        vars.insert("timbre".into(), "clarinet".into());
        let result = lib.expand("time_series_single", vars).unwrap();
        assert!(result.contains("data.csv"));
        assert!(result.contains("temperature"));
        assert!(result.contains("300Hz"));
        assert!(result.contains("clarinet"));
    }

    #[test]
    fn test_template_expand_alert() {
        let lib = TemplateLibrary::with_builtins();
        let mut vars = HashMap::new();
        vars.insert("alert_name".into(), "critical_temp".into());
        vars.insert("urgency".into(), "high".into());
        vars.insert("pitch".into(), "1200".into());
        vars.insert("repetitions".into(), "5".into());
        vars.insert("interval".into(), "0.2".into());
        vars.insert("escalate".into(), "true".into());
        let result = lib.expand("alert", vars).unwrap();
        assert!(result.contains("critical_temp"));
        assert!(result.contains("escalation = true"));
    }

    #[test]
    fn test_template_validate_ok() {
        let lib = TemplateLibrary::with_builtins();
        let mut vars = HashMap::new();
        vars.insert("pitch".into(), "880".into());
        vars.insert("repetitions".into(), "3".into());
        let errors = lib.validate("alert", &vars);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_template_validate_bad_type() {
        let lib = TemplateLibrary::with_builtins();
        let mut vars = HashMap::new();
        vars.insert("pitch".into(), "not_a_number".into());
        let errors = lib.validate("alert", &vars);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_template_library_custom() {
        let mut lib = TemplateLibrary::new();
        lib.register(TemplateDefinition {
            name: "custom".into(),
            description: "Custom template".into(),
            template: "value = {{val}}".into(),
            parameters: vec![
                TemplateParameter::new("val", ParamType::Float, "0", "A value"),
            ],
        });
        let mut vars = HashMap::new();
        vars.insert("val".into(), "42".into());
        let result = lib.expand("custom", vars).unwrap();
        assert_eq!(result, "value = 42");
    }

    #[test]
    fn test_template_expand_nonexistent() {
        let lib = TemplateLibrary::with_builtins();
        assert!(lib.expand("nonexistent", HashMap::new()).is_none());
    }
}

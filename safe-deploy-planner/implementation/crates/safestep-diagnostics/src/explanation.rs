//! Natural-language explanations for deployment plan decisions.
//!
//! Provides human-readable explanations of deployment plan steps, constraint
//! violations, infeasibility results, and points of no return. Supports
//! configurable verbosity levels and audience targeting.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during explanation generation.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum ExplanationError {
    #[error("missing parameter: {0}")]
    MissingParam(String),
    #[error("invalid template: {0}")]
    InvalidTemplate(String),
    #[error("causal chain is empty")]
    EmptyChain,
}

// ---------------------------------------------------------------------------
// VerbosityLevel / AudienceLevel
// ---------------------------------------------------------------------------

/// How much detail to include in generated explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerbosityLevel {
    Brief,
    Standard,
    Detailed,
    Technical,
}

impl Default for VerbosityLevel {
    fn default() -> Self {
        Self::Standard
    }
}

/// Target audience for generated explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AudienceLevel {
    Developer,
    Operator,
    Manager,
    Executive,
}

impl Default for AudienceLevel {
    fn default() -> Self {
        Self::Developer
    }
}

// ---------------------------------------------------------------------------
// ExplanationEngine
// ---------------------------------------------------------------------------

/// Central engine that produces human-readable explanations for plan decisions.
#[derive(Debug, Clone)]
pub struct ExplanationEngine {
    verbosity: VerbosityLevel,
    audience: AudienceLevel,
    nlg: NaturalLanguageGenerator,
}

impl Default for ExplanationEngine {
    fn default() -> Self {
        Self {
            verbosity: VerbosityLevel::Standard,
            audience: AudienceLevel::Developer,
            nlg: NaturalLanguageGenerator::new(),
        }
    }
}

impl ExplanationEngine {
    /// Create an engine configured with the given verbosity level.
    pub fn with_verbosity(level: VerbosityLevel) -> Self {
        Self {
            verbosity: level,
            ..Self::default()
        }
    }

    /// Create an engine configured with the given audience level.
    pub fn with_audience(level: AudienceLevel) -> Self {
        Self {
            audience: level,
            ..Self::default()
        }
    }

    /// Explain a Point of No Return.
    ///
    /// Expects JSON with optional fields: `state_id`, `services`, `reason`.
    pub fn explain_pnr(&self, pnr_info: &serde_json::Value) -> String {
        let state_id = pnr_info
            .get("state_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let services: Vec<&str> = pnr_info
            .get("services")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<&str>>()
            })
            .unwrap_or_default();
        let reason = pnr_info
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("rollback is no longer possible");

        match self.verbosity {
            VerbosityLevel::Brief => {
                format!("PNR at state {state_id}.")
            }
            VerbosityLevel::Standard => {
                if services.is_empty() {
                    format!(
                        "Point of no return at state {state_id}: {reason}.",
                    )
                } else {
                    format!(
                        "Point of no return at state {state_id} affecting {}: {reason}.",
                        services.join(", "),
                    )
                }
            }
            VerbosityLevel::Detailed => {
                let svc_text = if services.is_empty() {
                    String::from("no specific services listed")
                } else {
                    format!("services involved: {}", services.join(", "))
                };
                let audience_note = self.audience_note_pnr();
                format!(
                    "Point of no return at state {state_id}. {svc_text}. \
                     Reason: {reason}. {audience_note}",
                )
            }
            VerbosityLevel::Technical => {
                let svc_text = if services.is_empty() {
                    String::from("[]")
                } else {
                    format!("[{}]", services.join(", "))
                };
                format!(
                    "PNR(state={state_id}, services={svc_text}, reason=\"{reason}\"). \
                     Beyond this state the monotone deployment envelope \
                     cannot be reversed without data loss.",
                )
            }
        }
    }

    /// Explain why a set of constraints is infeasible.
    pub fn explain_infeasibility(&self, constraints: &[String]) -> String {
        if constraints.is_empty() {
            return "No conflicting constraints were identified.".to_string();
        }

        match self.verbosity {
            VerbosityLevel::Brief => {
                format!(
                    "Infeasible: {} conflicting constraint(s).",
                    constraints.len()
                )
            }
            VerbosityLevel::Standard => {
                let list = constraints
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("  {}. {c}", i + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                let intro = match self.audience {
                    AudienceLevel::Executive | AudienceLevel::Manager => {
                        "The deployment plan cannot proceed because the following requirements conflict"
                    }
                    _ => {
                        "The following constraints are mutually unsatisfiable"
                    }
                };
                format!("{intro}:\n{list}")
            }
            VerbosityLevel::Detailed => {
                let list = constraints
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("  {}. {c}", i + 1))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "Infeasibility detected: the constraint set has no satisfying assignment.\n\
                     Conflicting constraints ({} total):\n{list}\n\
                     Consider relaxing one or more constraints to make the plan feasible.",
                    constraints.len(),
                )
            }
            VerbosityLevel::Technical => {
                let list = constraints
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("  C{i}: {c}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "UNSAT core (size={}):\n{list}\n\
                     The SAT/SMT solver determined this subset is an irreducible infeasible set (IIS).",
                    constraints.len(),
                )
            }
        }
    }

    /// Explain a single constraint violation.
    pub fn explain_constraint_violation(
        &self,
        constraint_desc: &str,
        state_desc: &str,
    ) -> String {
        match self.verbosity {
            VerbosityLevel::Brief => {
                format!("Violation: {constraint_desc} in {state_desc}.")
            }
            VerbosityLevel::Standard => {
                let plain = self.nlg.technical_to_plain(constraint_desc);
                format!(
                    "Constraint violation detected in state {state_desc}: {plain}.",
                )
            }
            VerbosityLevel::Detailed => {
                let plain = self.nlg.technical_to_plain(constraint_desc);
                format!(
                    "A constraint was violated in deployment state {state_desc}.\n\
                     Constraint: {plain}.\n\
                     This violation must be resolved before the deployment can continue safely.",
                )
            }
            VerbosityLevel::Technical => {
                format!(
                    "VIOLATION(state={state_desc}, constraint=\"{constraint_desc}\"). \
                     The state lies outside the computed safety envelope.",
                )
            }
        }
    }

    /// Explain a single plan step.
    ///
    /// Expects JSON with optional fields: `service`, `from_version`, `to_version`,
    /// `duration_secs`, `prerequisites`.
    pub fn explain_plan_step(&self, step_info: &serde_json::Value) -> String {
        let service = step_info
            .get("service")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown-service");
        let from = step_info
            .get("from_version")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let to = step_info
            .get("to_version")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let duration = step_info
            .get("duration_secs")
            .and_then(|v| v.as_u64());
        let prereqs: Vec<&str> = step_info
            .get("prerequisites")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        match self.verbosity {
            VerbosityLevel::Brief => {
                format!("Update {service}: v{from} -> v{to}.")
            }
            VerbosityLevel::Standard => {
                let dur_text = duration
                    .map(|d| format!(" (estimated {d}s)"))
                    .unwrap_or_default();
                format!("Update service {service} from v{from} to v{to}{dur_text}.")
            }
            VerbosityLevel::Detailed => {
                let dur_text = duration
                    .map(|d| format!("Estimated duration: {d} seconds. ", ))
                    .unwrap_or_default();
                let prereq_text = if prereqs.is_empty() {
                    String::from("No prerequisites.")
                } else {
                    format!("Prerequisites: {}.", prereqs.join(", "))
                };
                format!(
                    "Deploy service {service}: upgrade from version {from} to version {to}. \
                     {dur_text}{prereq_text}",
                )
            }
            VerbosityLevel::Technical => {
                let prereq_list = if prereqs.is_empty() {
                    "[]".to_string()
                } else {
                    format!("[{}]", prereqs.join(", "))
                };
                format!(
                    "STEP(service={service}, from=v{from}, to=v{to}, \
                     duration={}, prereqs={prereq_list})",
                    duration
                        .map(|d| d.to_string())
                        .unwrap_or_else(|| "N/A".to_string()),
                )
            }
        }
    }

    fn audience_note_pnr(&self) -> &'static str {
        match self.audience {
            AudienceLevel::Developer => {
                "After this point, the deployment cannot be safely rolled back."
            }
            AudienceLevel::Operator => {
                "Manual intervention would be required to revert past this point."
            }
            AudienceLevel::Manager => {
                "Proceeding past this point commits the team to the new version."
            }
            AudienceLevel::Executive => {
                "This is a commitment point with no automatic undo capability."
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NaturalLanguageGenerator
// ---------------------------------------------------------------------------

/// Transforms templates and technical jargon into readable prose.
#[derive(Debug, Clone)]
pub struct NaturalLanguageGenerator {
    jargon_map: HashMap<String, String>,
}

impl NaturalLanguageGenerator {
    /// Create a new generator with the default jargon mapping.
    pub fn new() -> Self {
        let mut jargon_map = HashMap::new();
        jargon_map.insert("PNR".to_string(), "point of no return".to_string());
        jargon_map.insert("monotone".to_string(), "one-directional".to_string());
        jargon_map.insert("BMC".to_string(), "bounded model checking".to_string());
        jargon_map.insert("SAT".to_string(), "satisfiability".to_string());
        jargon_map.insert(
            "CEGAR".to_string(),
            "counterexample-guided refinement".to_string(),
        );
        jargon_map
            .insert("infeasible".to_string(), "impossible to achieve".to_string());
        jargon_map.insert("envelope".to_string(), "safe zone".to_string());
        Self { jargon_map }
    }

    /// Substitute `{param_name}` placeholders in `template` with values from
    /// `params`. Unknown placeholders are left as-is.
    pub fn generate(&self, template: &str, params: &HashMap<String, String>) -> String {
        let mut result = template.to_string();
        for (key, value) in params {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Replace known technical terms with plain-English equivalents.
    ///
    /// The replacement is case-sensitive and uses whole-word boundaries
    /// implemented via simple token splitting.
    pub fn technical_to_plain(&self, technical: &str) -> String {
        let mut result = technical.to_string();
        for (term, replacement) in &self.jargon_map {
            result = replace_whole_word(&result, term, replacement);
        }
        result
    }
}

impl Default for NaturalLanguageGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Replace `word` with `replacement` only when it appears as a whole word.
fn replace_whole_word(text: &str, word: &str, replacement: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let word_chars: Vec<char> = word.chars().collect();
    let word_len = word_chars.len();
    let text_len = chars.len();

    let mut i = 0;
    while i < text_len {
        if i + word_len <= text_len && chars[i..i + word_len] == word_chars[..] {
            let before_ok =
                i == 0 || !chars[i - 1].is_alphanumeric();
            let after_ok =
                i + word_len >= text_len || !chars[i + word_len].is_alphanumeric();
            if before_ok && after_ok {
                result.push_str(replacement);
                i += word_len;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

// ---------------------------------------------------------------------------
// ExplanationTemplate
// ---------------------------------------------------------------------------

/// A parameterised template that can be rendered with a set of values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplanationTemplate {
    pub template_str: String,
    pub required_params: Vec<String>,
}

impl ExplanationTemplate {
    /// Parse a template string, extracting `{param}` placeholders.
    pub fn new(template: &str) -> Self {
        let required_params = Self::extract_params(template);
        Self {
            template_str: template.to_string(),
            required_params,
        }
    }

    /// Validate that all required parameters are present.
    pub fn validate_params(
        &self,
        params: &HashMap<String, String>,
    ) -> Result<(), ExplanationError> {
        for p in &self.required_params {
            if !params.contains_key(p) {
                return Err(ExplanationError::MissingParam(p.clone()));
            }
        }
        Ok(())
    }

    /// Render the template by substituting all placeholders.
    pub fn render(
        &self,
        params: &HashMap<String, String>,
    ) -> Result<String, ExplanationError> {
        self.validate_params(params)?;
        let mut result = self.template_str.clone();
        for (key, value) in params {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, value);
        }
        Ok(result)
    }

    /// Common template for PNR explanations.
    pub fn pnr_template() -> Self {
        Self::new(
            "Point of no return at state {state_id}: after this point, \
             {reason}. Affected services: {services}.",
        )
    }

    /// Common template for step explanations.
    pub fn step_template() -> Self {
        Self::new("Update service {service} from v{from_version} to v{to_version}.")
    }

    /// Common template for constraint violation explanations.
    pub fn violation_template() -> Self {
        Self::new(
            "Constraint \"{constraint}\" violated in state {state}: {details}.",
        )
    }

    // -- internal helpers --

    fn extract_params(template: &str) -> Vec<String> {
        let mut params = Vec::new();
        let mut chars = template.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '{' {
                let mut name = String::new();
                for inner in chars.by_ref() {
                    if inner == '}' {
                        break;
                    }
                    name.push(inner);
                }
                if !name.is_empty() && !params.contains(&name) {
                    params.push(name);
                }
            }
        }
        params
    }
}

// ---------------------------------------------------------------------------
// CausalChain
// ---------------------------------------------------------------------------

/// A chain of cause→effect reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalChain {
    pub root_cause: String,
    pub intermediate: Vec<String>,
    pub final_effect: String,
}

impl CausalChain {
    /// Create a new chain with a root cause and final effect.
    pub fn new(root: &str, effect: &str) -> Self {
        Self {
            root_cause: root.to_string(),
            intermediate: Vec::new(),
            final_effect: effect.to_string(),
        }
    }

    /// Append an intermediate step between cause and effect.
    pub fn add_intermediate(&mut self, step: &str) {
        self.intermediate.push(step.to_string());
    }

    /// The total depth of the chain: root + intermediates + final effect.
    pub fn depth(&self) -> usize {
        2 + self.intermediate.len()
    }

    /// Build a new chain with the cause/effect direction reversed.
    pub fn reverse(&self) -> Self {
        let mut reversed_intermediates = self.intermediate.clone();
        reversed_intermediates.reverse();
        Self {
            root_cause: self.final_effect.clone(),
            intermediate: reversed_intermediates,
            final_effect: self.root_cause.clone(),
        }
    }

    /// Render the chain as a human-readable string.
    #[allow(clippy::inherent_to_string_shadow_display)]
    pub fn to_string(&self) -> String {
        let mut parts = Vec::with_capacity(self.depth());
        parts.push(self.root_cause.as_str());
        for step in &self.intermediate {
            parts.push(step.as_str());
        }
        parts.push(self.final_effect.as_str());

        match parts.len() {
            0 => String::new(),
            1 => parts[0].to_string(),
            2 => format!(
                "Because {}, resulting in {}",
                parts[0], parts[1]
            ),
            _ => {
                let first = parts[0];
                let last = parts[parts.len() - 1];
                let middle = &parts[1..parts.len() - 1];
                let middle_text = middle
                    .iter()
                    .map(|s| format!("which caused {s}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Because {first}, {middle_text}, resulting in {last}")
            }
        }
    }
}

impl fmt::Display for CausalChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ---------------------------------------------------------------------------
// Glossary
// ---------------------------------------------------------------------------

/// A glossary of deployment-related terms and their definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Glossary {
    pub terms: HashMap<String, String>,
}

impl Glossary {
    /// Create an empty glossary.
    pub fn new() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    /// Create a glossary pre-populated with common deployment terms.
    pub fn with_defaults() -> Self {
        let mut g = Self::new();
        g.define(
            "PNR",
            "Point of No Return — a state beyond which rollback is not possible.",
        );
        g.define(
            "envelope",
            "The set of deployment states proven safe by formal verification.",
        );
        g.define(
            "monotone",
            "A property meaning the deployment can only move forward, not backward.",
        );
        g.define(
            "BMC",
            "Bounded Model Checking — a verification technique that explores \
             states up to a fixed depth.",
        );
        g.define(
            "CEGAR",
            "Counter-Example Guided Abstraction Refinement — an iterative \
             verification approach.",
        );
        g.define(
            "SAT",
            "Satisfiability — determining whether a logical formula can be made true.",
        );
        g.define(
            "infeasible",
            "A plan or constraint set that has no valid solution.",
        );
        g.define(
            "rollback",
            "Reverting a deployment to a previous known-good state.",
        );
        g.define(
            "canary",
            "A deployment strategy that routes a small percentage of traffic \
             to the new version first.",
        );
        g.define(
            "blue-green",
            "A deployment strategy using two identical environments, \
             switching traffic between them.",
        );
        g.define(
            "service mesh",
            "Infrastructure layer that manages service-to-service communication.",
        );
        g.define(
            "health check",
            "An automated probe that verifies a service is functioning correctly.",
        );
        g
    }

    /// Define (or redefine) a term.
    pub fn define(&mut self, term: &str, definition: &str) {
        self.terms
            .insert(term.to_string(), definition.to_string());
    }

    /// Look up a term's definition.
    pub fn lookup(&self, term: &str) -> Option<&str> {
        self.terms.get(term).map(|s| s.as_str())
    }
}

impl Default for Glossary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ExplanationTemplate ----

    #[test]
    fn test_template_extract_params() {
        let tpl = ExplanationTemplate::new("Hello {name}, welcome to {place}!");
        assert_eq!(tpl.required_params, vec!["name", "place"]);
    }

    #[test]
    fn test_template_render_success() {
        let tpl = ExplanationTemplate::new("Service {svc} at v{ver}.");
        let mut params = HashMap::new();
        params.insert("svc".to_string(), "api".to_string());
        params.insert("ver".to_string(), "2.0".to_string());
        assert_eq!(tpl.render(&params).unwrap(), "Service api at v2.0.");
    }

    #[test]
    fn test_template_render_missing_param() {
        let tpl = ExplanationTemplate::new("{a} and {b}");
        let mut params = HashMap::new();
        params.insert("a".to_string(), "X".to_string());
        let err = tpl.render(&params).unwrap_err();
        assert_eq!(err, ExplanationError::MissingParam("b".to_string()));
    }

    #[test]
    fn test_template_validate_params_ok() {
        let tpl = ExplanationTemplate::new("{x}");
        let mut params = HashMap::new();
        params.insert("x".to_string(), "val".to_string());
        assert!(tpl.validate_params(&params).is_ok());
    }

    #[test]
    fn test_template_validate_params_missing() {
        let tpl = ExplanationTemplate::new("{x} {y}");
        let params = HashMap::new();
        assert!(tpl.validate_params(&params).is_err());
    }

    #[test]
    fn test_template_no_params() {
        let tpl = ExplanationTemplate::new("No placeholders here.");
        assert!(tpl.required_params.is_empty());
        assert_eq!(
            tpl.render(&HashMap::new()).unwrap(),
            "No placeholders here."
        );
    }

    #[test]
    fn test_pnr_template() {
        let tpl = ExplanationTemplate::pnr_template();
        assert!(tpl.required_params.contains(&"state_id".to_string()));
        assert!(tpl.required_params.contains(&"reason".to_string()));
        assert!(tpl.required_params.contains(&"services".to_string()));
    }

    #[test]
    fn test_step_template() {
        let tpl = ExplanationTemplate::step_template();
        let mut params = HashMap::new();
        params.insert("service".to_string(), "web".to_string());
        params.insert("from_version".to_string(), "1.0".to_string());
        params.insert("to_version".to_string(), "2.0".to_string());
        let rendered = tpl.render(&params).unwrap();
        assert_eq!(rendered, "Update service web from v1.0 to v2.0.");
    }

    #[test]
    fn test_violation_template() {
        let tpl = ExplanationTemplate::violation_template();
        assert!(tpl.required_params.contains(&"constraint".to_string()));
        assert!(tpl.required_params.contains(&"state".to_string()));
        assert!(tpl.required_params.contains(&"details".to_string()));
    }

    #[test]
    fn test_template_duplicate_params() {
        let tpl = ExplanationTemplate::new("{a} {b} {a}");
        assert_eq!(tpl.required_params, vec!["a", "b"]);
    }

    // ---- CausalChain ----

    #[test]
    fn test_causal_chain_simple() {
        let chain = CausalChain::new("root", "effect");
        assert_eq!(chain.depth(), 2);
        assert_eq!(chain.to_string(), "Because root, resulting in effect");
    }

    #[test]
    fn test_causal_chain_with_intermediates() {
        let mut chain = CausalChain::new("A", "D");
        chain.add_intermediate("B");
        chain.add_intermediate("C");
        assert_eq!(chain.depth(), 4);
        let s = chain.to_string();
        assert!(s.starts_with("Because A"));
        assert!(s.contains("which caused B"));
        assert!(s.contains("which caused C"));
        assert!(s.ends_with("resulting in D"));
    }

    #[test]
    fn test_causal_chain_display() {
        let chain = CausalChain::new("X", "Y");
        let display = format!("{chain}");
        assert_eq!(display, "Because X, resulting in Y");
    }

    #[test]
    fn test_causal_chain_reverse() {
        let mut chain = CausalChain::new("start", "end");
        chain.add_intermediate("mid1");
        chain.add_intermediate("mid2");
        let rev = chain.reverse();
        assert_eq!(rev.root_cause, "end");
        assert_eq!(rev.final_effect, "start");
        assert_eq!(rev.intermediate, vec!["mid2", "mid1"]);
    }

    #[test]
    fn test_causal_chain_single_intermediate() {
        let mut chain = CausalChain::new("cause", "effect");
        chain.add_intermediate("middle");
        let s = chain.to_string();
        assert_eq!(
            s,
            "Because cause, which caused middle, resulting in effect"
        );
    }

    #[test]
    fn test_causal_chain_reverse_no_intermediates() {
        let chain = CausalChain::new("A", "B");
        let rev = chain.reverse();
        assert_eq!(rev.root_cause, "B");
        assert_eq!(rev.final_effect, "A");
        assert!(rev.intermediate.is_empty());
    }

    #[test]
    fn test_causal_chain_depth_many() {
        let mut chain = CausalChain::new("root", "leaf");
        for i in 0..10 {
            chain.add_intermediate(&format!("step {i}"));
        }
        assert_eq!(chain.depth(), 12);
    }

    // ---- Glossary ----

    #[test]
    fn test_glossary_empty() {
        let g = Glossary::new();
        assert!(g.terms.is_empty());
        assert_eq!(g.lookup("PNR"), None);
    }

    #[test]
    fn test_glossary_define_and_lookup() {
        let mut g = Glossary::new();
        g.define("PNR", "Point of No Return");
        assert_eq!(g.lookup("PNR"), Some("Point of No Return"));
        assert_eq!(g.lookup("unknown"), None);
    }

    #[test]
    fn test_glossary_with_defaults() {
        let g = Glossary::with_defaults();
        assert!(g.lookup("PNR").is_some());
        assert!(g.lookup("envelope").is_some());
        assert!(g.lookup("monotone").is_some());
        assert!(g.lookup("BMC").is_some());
        assert!(g.lookup("CEGAR").is_some());
        assert!(g.lookup("SAT").is_some());
        assert!(g.lookup("infeasible").is_some());
        assert!(g.lookup("rollback").is_some());
        assert!(g.lookup("canary").is_some());
        assert!(g.lookup("blue-green").is_some());
        assert!(g.lookup("service mesh").is_some());
        assert!(g.lookup("health check").is_some());
    }

    #[test]
    fn test_glossary_redefine() {
        let mut g = Glossary::new();
        g.define("foo", "first");
        g.define("foo", "second");
        assert_eq!(g.lookup("foo"), Some("second"));
    }

    #[test]
    fn test_glossary_default_count() {
        let g = Glossary::with_defaults();
        assert_eq!(g.terms.len(), 12);
    }

    // ---- NaturalLanguageGenerator ----

    #[test]
    fn test_nlg_generate_basic() {
        let nlg = NaturalLanguageGenerator::new();
        let mut params = HashMap::new();
        params.insert("name".to_string(), "Alice".to_string());
        assert_eq!(nlg.generate("Hello {name}!", &params), "Hello Alice!");
    }

    #[test]
    fn test_nlg_generate_multiple_params() {
        let nlg = NaturalLanguageGenerator::new();
        let mut params = HashMap::new();
        params.insert("a".to_string(), "X".to_string());
        params.insert("b".to_string(), "Y".to_string());
        assert_eq!(nlg.generate("{a} and {b}", &params), "X and Y");
    }

    #[test]
    fn test_nlg_generate_no_match() {
        let nlg = NaturalLanguageGenerator::new();
        let params = HashMap::new();
        assert_eq!(
            nlg.generate("No {placeholders} here", &params),
            "No {placeholders} here"
        );
    }

    #[test]
    fn test_nlg_technical_to_plain_pnr() {
        let nlg = NaturalLanguageGenerator::new();
        let result = nlg.technical_to_plain("PNR detected");
        assert_eq!(result, "point of no return detected");
    }

    #[test]
    fn test_nlg_technical_to_plain_multiple() {
        let nlg = NaturalLanguageGenerator::new();
        let result = nlg.technical_to_plain("The SAT solver found the plan infeasible");
        assert!(result.contains("satisfiability"));
        assert!(result.contains("impossible to achieve"));
    }

    #[test]
    fn test_nlg_technical_to_plain_envelope() {
        let nlg = NaturalLanguageGenerator::new();
        let result = nlg.technical_to_plain("Outside the envelope");
        assert_eq!(result, "Outside the safe zone");
    }

    #[test]
    fn test_nlg_technical_to_plain_bmc() {
        let nlg = NaturalLanguageGenerator::new();
        let result = nlg.technical_to_plain("BMC depth exceeded");
        assert_eq!(result, "bounded model checking depth exceeded");
    }

    #[test]
    fn test_nlg_technical_to_plain_cegar() {
        let nlg = NaturalLanguageGenerator::new();
        let result = nlg.technical_to_plain("CEGAR loop");
        assert_eq!(result, "counterexample-guided refinement loop");
    }

    #[test]
    fn test_nlg_technical_to_plain_no_jargon() {
        let nlg = NaturalLanguageGenerator::new();
        let input = "everything is fine";
        assert_eq!(nlg.technical_to_plain(input), input);
    }

    #[test]
    fn test_nlg_technical_to_plain_monotone() {
        let nlg = NaturalLanguageGenerator::new();
        let result = nlg.technical_to_plain("monotone property");
        assert_eq!(result, "one-directional property");
    }

    // ---- ExplanationEngine ----

    #[test]
    fn test_engine_default() {
        let engine = ExplanationEngine::default();
        assert_eq!(engine.verbosity, VerbosityLevel::Standard);
        assert_eq!(engine.audience, AudienceLevel::Developer);
    }

    #[test]
    fn test_engine_with_verbosity() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Brief);
        assert_eq!(engine.verbosity, VerbosityLevel::Brief);
    }

    #[test]
    fn test_engine_with_audience() {
        let engine = ExplanationEngine::with_audience(AudienceLevel::Executive);
        assert_eq!(engine.audience, AudienceLevel::Executive);
    }

    #[test]
    fn test_explain_pnr_brief() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Brief);
        let info = serde_json::json!({"state_id": "S5"});
        let result = engine.explain_pnr(&info);
        assert_eq!(result, "PNR at state S5.");
    }

    #[test]
    fn test_explain_pnr_standard_no_services() {
        let engine = ExplanationEngine::default();
        let info = serde_json::json!({
            "state_id": "S3",
            "reason": "database migration applied"
        });
        let result = engine.explain_pnr(&info);
        assert!(result.contains("Point of no return at state S3"));
        assert!(result.contains("database migration applied"));
    }

    #[test]
    fn test_explain_pnr_standard_with_services() {
        let engine = ExplanationEngine::default();
        let info = serde_json::json!({
            "state_id": "S3",
            "services": ["api", "web"],
            "reason": "schema changed"
        });
        let result = engine.explain_pnr(&info);
        assert!(result.contains("api, web"));
        assert!(result.contains("schema changed"));
    }

    #[test]
    fn test_explain_pnr_detailed() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Detailed);
        let info = serde_json::json!({
            "state_id": "S7",
            "services": ["db"],
            "reason": "data migration"
        });
        let result = engine.explain_pnr(&info);
        assert!(result.contains("Point of no return at state S7"));
        assert!(result.contains("services involved: db"));
        assert!(result.contains("data migration"));
        assert!(result.contains("rolled back"));
    }

    #[test]
    fn test_explain_pnr_technical() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Technical);
        let info = serde_json::json!({
            "state_id": "S1",
            "services": ["svc-a"],
            "reason": "irreversible"
        });
        let result = engine.explain_pnr(&info);
        assert!(result.contains("PNR(state=S1"));
        assert!(result.contains("monotone"));
    }

    #[test]
    fn test_explain_infeasibility_empty() {
        let engine = ExplanationEngine::default();
        let result = engine.explain_infeasibility(&[]);
        assert_eq!(result, "No conflicting constraints were identified.");
    }

    #[test]
    fn test_explain_infeasibility_standard() {
        let engine = ExplanationEngine::default();
        let constraints = vec![
            "api >= 2.0".to_string(),
            "web requires api < 2.0".to_string(),
        ];
        let result = engine.explain_infeasibility(&constraints);
        assert!(result.contains("unsatisfiable"));
        assert!(result.contains("api >= 2.0"));
        assert!(result.contains("web requires api < 2.0"));
    }

    #[test]
    fn test_explain_infeasibility_brief() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Brief);
        let constraints = vec!["c1".to_string(), "c2".to_string(), "c3".to_string()];
        let result = engine.explain_infeasibility(&constraints);
        assert_eq!(result, "Infeasible: 3 conflicting constraint(s).");
    }

    #[test]
    fn test_explain_infeasibility_detailed() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Detailed);
        let constraints = vec!["X".to_string()];
        let result = engine.explain_infeasibility(&constraints);
        assert!(result.contains("Infeasibility detected"));
        assert!(result.contains("1 total"));
        assert!(result.contains("Consider relaxing"));
    }

    #[test]
    fn test_explain_infeasibility_technical() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Technical);
        let constraints = vec!["c1".to_string()];
        let result = engine.explain_infeasibility(&constraints);
        assert!(result.contains("UNSAT core"));
        assert!(result.contains("IIS"));
    }

    #[test]
    fn test_explain_infeasibility_manager_audience() {
        let engine = ExplanationEngine {
            verbosity: VerbosityLevel::Standard,
            audience: AudienceLevel::Manager,
            nlg: NaturalLanguageGenerator::new(),
        };
        let constraints = vec!["c1".to_string()];
        let result = engine.explain_infeasibility(&constraints);
        assert!(result.contains("cannot proceed"));
    }

    #[test]
    fn test_explain_constraint_violation_brief() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Brief);
        let result =
            engine.explain_constraint_violation("version compat", "S2");
        assert_eq!(result, "Violation: version compat in S2.");
    }

    #[test]
    fn test_explain_constraint_violation_standard() {
        let engine = ExplanationEngine::default();
        let result =
            engine.explain_constraint_violation("PNR ordering", "S4");
        let expected_contains = "point of no return";
        assert!(result.contains(expected_contains));
    }

    #[test]
    fn test_explain_constraint_violation_detailed() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Detailed);
        let result =
            engine.explain_constraint_violation("resource cap", "S9");
        assert!(result.contains("constraint was violated"));
        assert!(result.contains("S9"));
        assert!(result.contains("resolved"));
    }

    #[test]
    fn test_explain_constraint_violation_technical() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Technical);
        let result =
            engine.explain_constraint_violation("dep(a,b)", "S0");
        assert!(result.contains("VIOLATION"));
        assert!(result.contains("safety envelope"));
    }

    #[test]
    fn test_explain_plan_step_brief() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Brief);
        let info = serde_json::json!({
            "service": "api",
            "from_version": "1.0.0",
            "to_version": "2.0.0"
        });
        let result = engine.explain_plan_step(&info);
        assert_eq!(result, "Update api: v1.0.0 -> v2.0.0.");
    }

    #[test]
    fn test_explain_plan_step_standard() {
        let engine = ExplanationEngine::default();
        let info = serde_json::json!({
            "service": "web",
            "from_version": "1.0",
            "to_version": "1.1",
            "duration_secs": 45
        });
        let result = engine.explain_plan_step(&info);
        assert!(result.contains("Update service web"));
        assert!(result.contains("estimated 45s"));
    }

    #[test]
    fn test_explain_plan_step_detailed() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Detailed);
        let info = serde_json::json!({
            "service": "db",
            "from_version": "3.0",
            "to_version": "4.0",
            "duration_secs": 120,
            "prerequisites": ["step-1", "step-2"]
        });
        let result = engine.explain_plan_step(&info);
        assert!(result.contains("Deploy service db"));
        assert!(result.contains("120 seconds"));
        assert!(result.contains("step-1, step-2"));
    }

    #[test]
    fn test_explain_plan_step_detailed_no_prereqs() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Detailed);
        let info = serde_json::json!({
            "service": "cache",
            "from_version": "1.0",
            "to_version": "2.0"
        });
        let result = engine.explain_plan_step(&info);
        assert!(result.contains("No prerequisites"));
    }

    #[test]
    fn test_explain_plan_step_technical() {
        let engine = ExplanationEngine::with_verbosity(VerbosityLevel::Technical);
        let info = serde_json::json!({
            "service": "api",
            "from_version": "1.0",
            "to_version": "2.0",
            "duration_secs": 30,
            "prerequisites": ["s0"]
        });
        let result = engine.explain_plan_step(&info);
        assert!(result.starts_with("STEP("));
        assert!(result.contains("prereqs=[s0]"));
    }

    #[test]
    fn test_explain_plan_step_missing_fields() {
        let engine = ExplanationEngine::default();
        let info = serde_json::json!({});
        let result = engine.explain_plan_step(&info);
        assert!(result.contains("unknown-service"));
    }

    // ---- Edge cases ----

    #[test]
    fn test_template_empty_string() {
        let tpl = ExplanationTemplate::new("");
        assert!(tpl.required_params.is_empty());
        assert_eq!(tpl.render(&HashMap::new()).unwrap(), "");
    }

    #[test]
    fn test_template_only_placeholder() {
        let tpl = ExplanationTemplate::new("{x}");
        let mut params = HashMap::new();
        params.insert("x".to_string(), "VALUE".to_string());
        assert_eq!(tpl.render(&params).unwrap(), "VALUE");
    }

    #[test]
    fn test_nlg_generate_empty_template() {
        let nlg = NaturalLanguageGenerator::new();
        assert_eq!(nlg.generate("", &HashMap::new()), "");
    }

    #[test]
    fn test_replace_whole_word_no_partial() {
        let result = replace_whole_word("SATISFACTION", "SAT", "satisfiability");
        assert_eq!(result, "SATISFACTION");
    }

    #[test]
    fn test_replace_whole_word_at_boundaries() {
        let result = replace_whole_word("SAT is hard", "SAT", "satisfiability");
        assert_eq!(result, "satisfiability is hard");
    }

    #[test]
    fn test_causal_chain_serialize_roundtrip() {
        let mut chain = CausalChain::new("cause", "effect");
        chain.add_intermediate("mid");
        let json = serde_json::to_string(&chain).unwrap();
        let deser: CausalChain = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, chain);
    }

    #[test]
    fn test_verbosity_level_serialize() {
        let json = serde_json::to_string(&VerbosityLevel::Technical).unwrap();
        let deser: VerbosityLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, VerbosityLevel::Technical);
    }

    #[test]
    fn test_audience_level_serialize() {
        let json = serde_json::to_string(&AudienceLevel::Executive).unwrap();
        let deser: AudienceLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, AudienceLevel::Executive);
    }

    #[test]
    fn test_glossary_serialize_roundtrip() {
        let g = Glossary::with_defaults();
        let json = serde_json::to_string(&g).unwrap();
        let deser: Glossary = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.terms.len(), g.terms.len());
    }

    #[test]
    fn test_template_serialize_roundtrip() {
        let tpl = ExplanationTemplate::pnr_template();
        let json = serde_json::to_string(&tpl).unwrap();
        let deser: ExplanationTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, tpl);
    }

    #[test]
    fn test_explain_pnr_empty_json() {
        let engine = ExplanationEngine::default();
        let info = serde_json::json!({});
        let result = engine.explain_pnr(&info);
        assert!(result.contains("unknown"));
    }

    #[test]
    fn test_engine_audience_detailed_pnr_developer() {
        let engine = ExplanationEngine {
            verbosity: VerbosityLevel::Detailed,
            audience: AudienceLevel::Developer,
            nlg: NaturalLanguageGenerator::new(),
        };
        let info = serde_json::json!({"state_id": "S1"});
        let result = engine.explain_pnr(&info);
        assert!(result.contains("rolled back"));
    }

    #[test]
    fn test_engine_audience_detailed_pnr_executive() {
        let engine = ExplanationEngine {
            verbosity: VerbosityLevel::Detailed,
            audience: AudienceLevel::Executive,
            nlg: NaturalLanguageGenerator::new(),
        };
        let info = serde_json::json!({"state_id": "S1"});
        let result = engine.explain_pnr(&info);
        assert!(result.contains("commitment point"));
    }

    #[test]
    fn test_engine_audience_detailed_pnr_operator() {
        let engine = ExplanationEngine {
            verbosity: VerbosityLevel::Detailed,
            audience: AudienceLevel::Operator,
            nlg: NaturalLanguageGenerator::new(),
        };
        let info = serde_json::json!({"state_id": "S1"});
        let result = engine.explain_pnr(&info);
        assert!(result.contains("Manual intervention"));
    }

    #[test]
    fn test_engine_audience_detailed_pnr_manager() {
        let engine = ExplanationEngine {
            verbosity: VerbosityLevel::Detailed,
            audience: AudienceLevel::Manager,
            nlg: NaturalLanguageGenerator::new(),
        };
        let info = serde_json::json!({"state_id": "S1"});
        let result = engine.explain_pnr(&info);
        assert!(result.contains("commits the team"));
    }

    #[test]
    fn test_template_render_extra_params_ok() {
        let tpl = ExplanationTemplate::new("{a}");
        let mut params = HashMap::new();
        params.insert("a".to_string(), "val".to_string());
        params.insert("extra".to_string(), "ignored".to_string());
        assert_eq!(tpl.render(&params).unwrap(), "val");
    }

    #[test]
    fn test_replace_whole_word_end_of_string() {
        let result = replace_whole_word("check SAT", "SAT", "satisfiability");
        assert_eq!(result, "check satisfiability");
    }

    #[test]
    fn test_replace_whole_word_entire_string() {
        let result = replace_whole_word("SAT", "SAT", "satisfiability");
        assert_eq!(result, "satisfiability");
    }
}

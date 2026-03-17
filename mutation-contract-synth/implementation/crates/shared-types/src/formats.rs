//! Contract output format support.
//!
//! Provides formatters for emitting contracts in multiple output formats:
//! JML annotations, Rust `contracts` crate attributes, machine-readable JSON,
//! SARIF integration types, and plain text.

use std::fmt;

use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::contracts::{Contract, ContractClause, ContractStrength, Specification};
use crate::types::{FunctionSignature, QfLiaType};

// ---------------------------------------------------------------------------
// ContractFormat
// ---------------------------------------------------------------------------

/// Supported output formats for contract rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContractFormat {
    /// JML-style annotations (`/*@ ... @*/`).
    Jml,
    /// Rust `contracts` crate attributes (`#[requires(...)]`).
    RustContracts,
    /// Machine-readable JSON with metadata.
    Json,
    /// SARIF-compatible result embedding.
    Sarif,
    /// Plain text (the `Display` representation).
    Text,
}

impl ContractFormat {
    pub fn name(&self) -> &'static str {
        match self {
            ContractFormat::Jml => "JML",
            ContractFormat::RustContracts => "Rust Contracts",
            ContractFormat::Json => "JSON",
            ContractFormat::Sarif => "SARIF",
            ContractFormat::Text => "Text",
        }
    }

    pub fn file_extension(&self) -> &'static str {
        match self {
            ContractFormat::Jml => "jml",
            ContractFormat::RustContracts => "rs",
            ContractFormat::Json => "json",
            ContractFormat::Sarif => "sarif",
            ContractFormat::Text => "txt",
        }
    }
}

impl fmt::Display for ContractFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// FormatOptions
// ---------------------------------------------------------------------------

/// Options controlling the output of contract formatters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatOptions {
    /// Include provenance comments in the output.
    pub include_provenance: bool,
    /// Include strength annotations.
    pub include_strength: bool,
    /// Indentation string (e.g. `"    "` or `"\t"`).
    pub indent: String,
    /// Tool version string for JSON/SARIF metadata.
    pub tool_version: String,
    /// Whether to emit `@spec_public` / `@pure` JML annotations.
    pub jml_visibility_annotations: bool,
    /// Function signatures to use when generating full annotated output.
    pub signatures: Vec<FunctionSignature>,
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            include_provenance: false,
            include_strength: true,
            indent: "    ".to_string(),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            jml_visibility_annotations: true,
            signatures: Vec::new(),
        }
    }
}

impl FormatOptions {
    /// Find the signature for a given function name, if provided.
    pub fn signature_for(&self, name: &str) -> Option<&FunctionSignature> {
        self.signatures.iter().find(|s| s.name == name)
    }
}

// ---------------------------------------------------------------------------
// ContractFormatter trait
// ---------------------------------------------------------------------------

/// Trait for rendering contracts in a specific output format.
pub trait ContractFormatter {
    /// Render a single [`Contract`] to a string.
    fn format_contract(&self, contract: &Contract, opts: &FormatOptions) -> String;

    /// Render an entire [`Specification`] to a string.
    fn format_specification(&self, spec: &Specification, opts: &FormatOptions) -> String;
}

/// Create a boxed formatter for the given format.
pub fn formatter_for(format: ContractFormat) -> Box<dyn ContractFormatter> {
    match format {
        ContractFormat::Jml => Box::new(JmlFormatter),
        ContractFormat::RustContracts => Box::new(RustContractsFormatter),
        ContractFormat::Json => Box::new(JsonContractFormatter),
        ContractFormat::Sarif => Box::new(SarifFormatter),
        ContractFormat::Text => Box::new(TextFormatter),
    }
}

// ===========================================================================
// JmlFormatter
// ===========================================================================

/// Formats contracts as JML annotation blocks using `/*@ ... @*/` syntax.
pub struct JmlFormatter;

impl JmlFormatter {
    /// Format a single clause as a JML annotation line (without block wrapper).
    fn format_clause(clause: &ContractClause, indent: &str) -> String {
        match clause {
            ContractClause::Requires(f) => format!("{indent}@ requires {f};"),
            ContractClause::Ensures(f) => format!("{indent}@ ensures {f};"),
            ContractClause::Invariant(f) => format!("{indent}@ loop_invariant {f};"),
        }
    }

    /// Render the `@assignable` clause (defaults to `\nothing` when we have no
    /// mutation information suggesting otherwise).
    fn assignable_clause(indent: &str) -> String {
        format!("{indent}@ assignable \\nothing;")
    }

    /// Render `@signals` clause for unchecked exceptions.
    fn signals_clause(indent: &str) -> String {
        format!("{indent}@ signals (RuntimeException e) false;")
    }

    /// Render a full JML block for a contract, optionally with a signature.
    pub fn format_block(
        contract: &Contract,
        opts: &FormatOptions,
        sig: Option<&FunctionSignature>,
    ) -> String {
        let ind = &opts.indent;
        let mut lines: Vec<String> = Vec::new();

        // Visibility / purity annotations before the block.
        if opts.jml_visibility_annotations {
            if let Some(s) = sig {
                lines.push(format!("{ind}//@ spec_public"));
                if s.return_type != QfLiaType::Void {
                    lines.push(format!("{ind}//@ pure"));
                }
            }
        }

        // Block-style annotation.
        lines.push(format!("{ind}/*@"));

        // Preconditions.
        for clause in &contract.clauses {
            if clause.is_requires() {
                lines.push(Self::format_clause(clause, ind));
            }
        }

        // Postconditions.
        for clause in &contract.clauses {
            if clause.is_ensures() {
                lines.push(Self::format_clause(clause, ind));
            }
        }

        // Invariants.
        for clause in &contract.clauses {
            if clause.is_invariant() {
                lines.push(Self::format_clause(clause, ind));
            }
        }

        // Assignable + signals.
        lines.push(Self::assignable_clause(ind));
        lines.push(Self::signals_clause(ind));

        if opts.include_strength {
            lines.push(format!("{ind}@ // strength: {}", contract.strength));
        }

        lines.push(format!("{ind}@*/"));

        lines.join("\n")
    }
}

impl ContractFormatter for JmlFormatter {
    fn format_contract(&self, contract: &Contract, opts: &FormatOptions) -> String {
        let sig = opts.signature_for(&contract.function_name);
        Self::format_block(contract, opts, sig)
    }

    fn format_specification(&self, spec: &Specification, opts: &FormatOptions) -> String {
        spec.contracts
            .iter()
            .map(|c| self.format_contract(c, opts))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

// ===========================================================================
// RustContractsFormatter
// ===========================================================================

/// Formats contracts as Rust `contracts` crate attributes.
pub struct RustContractsFormatter;

impl RustContractsFormatter {
    /// Map a QF-LIA type to its Rust equivalent.
    fn rust_type(ty: &QfLiaType) -> &'static str {
        match ty {
            QfLiaType::Int => "i32",
            QfLiaType::Long => "i64",
            QfLiaType::Boolean => "bool",
            QfLiaType::IntArray => "&[i32]",
            QfLiaType::Void => "()",
        }
    }

    /// Render a Rust function signature line.
    fn rust_signature(sig: &FunctionSignature) -> String {
        let params: Vec<String> = sig
            .params
            .iter()
            .map(|(name, ty)| format!("{name}: {}", Self::rust_type(ty)))
            .collect();
        let ret = if sig.return_type == QfLiaType::Void {
            String::new()
        } else {
            format!(" -> {}", Self::rust_type(&sig.return_type))
        };
        format!("fn {}({}){ret}", sig.name, params.join(", "))
    }

    /// Render a clause as a Rust attribute.  For `ensures` clauses the result
    /// variable is referenced as `ret`.
    fn format_clause(clause: &ContractClause) -> String {
        match clause {
            ContractClause::Requires(f) => format!("#[requires({f})]"),
            ContractClause::Ensures(f) => {
                let expr = format!("{f}");
                let expr = expr.replace("\\result", "ret");
                format!("#[ensures(ret -> {expr})]")
            }
            ContractClause::Invariant(f) => format!("#[invariant({f})]"),
        }
    }
}

impl ContractFormatter for RustContractsFormatter {
    fn format_contract(&self, contract: &Contract, opts: &FormatOptions) -> String {
        let mut lines: Vec<String> = Vec::new();

        if opts.include_strength {
            lines.push(format!("// Contract strength: {}", contract.strength));
        }

        for clause in &contract.clauses {
            lines.push(Self::format_clause(clause));
        }

        if let Some(sig) = opts.signature_for(&contract.function_name) {
            lines.push(format!("pub {} {{ todo!() }}", Self::rust_signature(sig)));
        }

        lines.join("\n")
    }

    fn format_specification(&self, spec: &Specification, opts: &FormatOptions) -> String {
        spec.contracts
            .iter()
            .map(|c| self.format_contract(c, opts))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

// ===========================================================================
// JsonContractFormatter
// ===========================================================================

/// Formats contracts as machine-readable JSON.
pub struct JsonContractFormatter;

/// JSON-serializable representation of a full contract export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonContractOutput {
    /// Schema version.
    pub schema_version: String,
    /// Metadata about the tool run.
    pub metadata: JsonMetadata,
    /// Exported contracts.
    pub contracts: Vec<JsonContractEntry>,
}

/// Metadata embedded in the JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonMetadata {
    pub tool: String,
    pub version: String,
    pub timestamp: String,
}

/// A single contract in JSON form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonContractEntry {
    pub function_name: String,
    pub strength: String,
    pub verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<JsonSignature>,
    pub clauses: Vec<JsonClauseEntry>,
    pub provenance: Vec<JsonProvenanceEntry>,
}

/// Function signature in JSON form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonSignature {
    pub name: String,
    pub parameters: Vec<JsonParam>,
    pub return_type: String,
}

/// A parameter in JSON form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonParam {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
}

/// A clause in JSON form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonClauseEntry {
    pub kind: String,
    pub formula: String,
}

/// Provenance entry in JSON form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonProvenanceEntry {
    pub tier: String,
    pub mutant_count: usize,
    pub solver_queries: u32,
    pub synthesis_time_ms: f64,
}

impl JsonContractFormatter {
    fn build_output(spec: &Specification, opts: &FormatOptions) -> JsonContractOutput {
        let contracts = spec
            .contracts
            .iter()
            .map(|c| Self::build_entry(c, opts))
            .collect();

        JsonContractOutput {
            schema_version: "1.0.0".to_string(),
            metadata: JsonMetadata {
                tool: "MutSpec".to_string(),
                version: opts.tool_version.clone(),
                timestamp: Utc::now().to_rfc3339(),
            },
            contracts,
        }
    }

    fn build_entry(contract: &Contract, opts: &FormatOptions) -> JsonContractEntry {
        let sig = opts
            .signature_for(&contract.function_name)
            .map(|s| JsonSignature {
                name: s.name.clone(),
                parameters: s
                    .params
                    .iter()
                    .map(|(n, ty)| JsonParam {
                        name: n.clone(),
                        ty: ty.name().to_string(),
                    })
                    .collect(),
                return_type: s.return_type.name().to_string(),
            });

        let clauses = contract
            .clauses
            .iter()
            .map(|c| JsonClauseEntry {
                kind: c.kind_name().to_string(),
                formula: format!("{}", c.formula()),
            })
            .collect();

        let provenance = contract
            .provenance
            .iter()
            .map(|p| JsonProvenanceEntry {
                tier: p.tier.name().to_string(),
                mutant_count: p.mutant_count(),
                solver_queries: p.solver_queries,
                synthesis_time_ms: p.synthesis_time_ms,
            })
            .collect();

        JsonContractEntry {
            function_name: contract.function_name.clone(),
            strength: contract.strength.name().to_string(),
            verified: contract.verified,
            signature: sig,
            clauses,
            provenance,
        }
    }
}

impl ContractFormatter for JsonContractFormatter {
    fn format_contract(&self, contract: &Contract, opts: &FormatOptions) -> String {
        let entry = Self::build_entry(contract, opts);
        serde_json::to_string_pretty(&entry).unwrap_or_default()
    }

    fn format_specification(&self, spec: &Specification, opts: &FormatOptions) -> String {
        let output = Self::build_output(spec, opts);
        serde_json::to_string_pretty(&output).unwrap_or_default()
    }
}

// ===========================================================================
// SARIF integration types
// ===========================================================================

/// A SARIF rule definition for a contract violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifContractRule {
    /// Stable rule identifier (e.g. `"mutspec/requires-violation"`).
    pub id: String,
    /// Short description of the rule.
    pub short_description: String,
    /// Longer help text.
    pub full_description: String,
    /// Default severity level.
    pub default_level: SarifLevel,
}

/// SARIF severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SarifLevel {
    Error,
    Warning,
    Note,
    None,
}

impl fmt::Display for SarifLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SarifLevel::Error => f.write_str("error"),
            SarifLevel::Warning => f.write_str("warning"),
            SarifLevel::Note => f.write_str("note"),
            SarifLevel::None => f.write_str("none"),
        }
    }
}

/// A SARIF result embedding contract information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifContractResult {
    pub rule_id: String,
    pub level: SarifLevel,
    pub message: String,
    pub function_name: String,
    pub clause_kind: String,
    pub formula: String,
}

/// Predefined SARIF rule definitions for contract violations.
pub fn sarif_contract_rules() -> Vec<SarifContractRule> {
    vec![
        SarifContractRule {
            id: "mutspec/requires-violation".to_string(),
            short_description: "Precondition may not hold at call site".to_string(),
            full_description: "A synthesised precondition was not satisfied at one or more \
                               call sites, indicating a potential bug."
                .to_string(),
            default_level: SarifLevel::Error,
        },
        SarifContractRule {
            id: "mutspec/ensures-violation".to_string(),
            short_description: "Postcondition may not hold on function exit".to_string(),
            full_description: "A synthesised postcondition could not be verified for every \
                               execution path, indicating a potential implementation defect."
                .to_string(),
            default_level: SarifLevel::Error,
        },
        SarifContractRule {
            id: "mutspec/invariant-violation".to_string(),
            short_description: "Loop invariant may not be maintained".to_string(),
            full_description: "A synthesised loop invariant was not preserved across an \
                               iteration, indicating a potential logic error."
                .to_string(),
            default_level: SarifLevel::Warning,
        },
        SarifContractRule {
            id: "mutspec/weak-contract".to_string(),
            short_description: "Contract is too weak to kill surviving mutants".to_string(),
            full_description: "The synthesised contract does not kill a sufficient fraction \
                               of generated mutants; the specification may be incomplete."
                .to_string(),
            default_level: SarifLevel::Note,
        },
    ]
}

/// Formats contracts as a SARIF-compatible JSON result array.
pub struct SarifFormatter;

impl SarifFormatter {
    fn build_results(contract: &Contract) -> Vec<SarifContractResult> {
        let mut results = Vec::new();

        for clause in &contract.clauses {
            let (rule_id, kind) = match clause {
                ContractClause::Requires(_) => ("mutspec/requires-violation", "requires"),
                ContractClause::Ensures(_) => ("mutspec/ensures-violation", "ensures"),
                ContractClause::Invariant(_) => ("mutspec/invariant-violation", "invariant"),
            };

            results.push(SarifContractResult {
                rule_id: rule_id.to_string(),
                level: SarifLevel::Warning,
                message: format!(
                    "{kind} clause for `{}`: {}",
                    contract.function_name,
                    clause.formula()
                ),
                function_name: contract.function_name.clone(),
                clause_kind: kind.to_string(),
                formula: format!("{}", clause.formula()),
            });
        }

        if contract.strength == ContractStrength::Weak
            || contract.strength == ContractStrength::Trivial
        {
            results.push(SarifContractResult {
                rule_id: "mutspec/weak-contract".to_string(),
                level: SarifLevel::Note,
                message: format!(
                    "Contract for `{}` has strength '{}'",
                    contract.function_name, contract.strength
                ),
                function_name: contract.function_name.clone(),
                clause_kind: "meta".to_string(),
                formula: String::new(),
            });
        }

        results
    }
}

/// Top-level SARIF output wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifOutput {
    pub version: String,
    pub tool: SarifToolInfo,
    pub rules: Vec<SarifContractRule>,
    pub results: Vec<SarifContractResult>,
}

/// SARIF tool descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarifToolInfo {
    pub name: String,
    pub version: String,
}

impl ContractFormatter for SarifFormatter {
    fn format_contract(&self, contract: &Contract, _opts: &FormatOptions) -> String {
        let results = Self::build_results(contract);
        serde_json::to_string_pretty(&results).unwrap_or_default()
    }

    fn format_specification(&self, spec: &Specification, opts: &FormatOptions) -> String {
        let results: Vec<SarifContractResult> = spec
            .contracts
            .iter()
            .flat_map(|c| Self::build_results(c))
            .collect();

        let output = SarifOutput {
            version: "2.1.0".to_string(),
            tool: SarifToolInfo {
                name: "MutSpec".to_string(),
                version: opts.tool_version.clone(),
            },
            rules: sarif_contract_rules(),
            results,
        };

        serde_json::to_string_pretty(&output).unwrap_or_default()
    }
}

// ===========================================================================
// TextFormatter
// ===========================================================================

/// Formats contracts using the built-in `Display` implementation.
pub struct TextFormatter;

impl ContractFormatter for TextFormatter {
    fn format_contract(&self, contract: &Contract, _opts: &FormatOptions) -> String {
        contract.to_string()
    }

    fn format_specification(&self, spec: &Specification, _opts: &FormatOptions) -> String {
        spec.to_string()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::{Contract, ContractClause, ContractProvenance, SynthesisTier};
    use crate::formula::{Formula, Predicate, Relation, Term};
    use crate::types::{FunctionSignature, QfLiaType};

    fn sample_contract() -> Contract {
        let pre = Formula::Atom(Predicate {
            relation: Relation::Gt,
            left: Term::Var("x".into()),
            right: Term::Const(0),
        });
        let post = Formula::Atom(Predicate {
            relation: Relation::Ge,
            left: Term::Var("\\result".into()),
            right: Term::Const(1),
        });

        let mut c = Contract::new("compute");
        c.add_clause(ContractClause::requires(pre));
        c.add_clause(ContractClause::ensures(post));
        c.set_strength(ContractStrength::Adequate);
        c.add_provenance(
            ContractProvenance::new(SynthesisTier::Tier1LatticeWalk)
                .with_solver_queries(3)
                .with_time(12.5),
        );
        c
    }

    fn sample_sig() -> FunctionSignature {
        FunctionSignature::new(
            "compute",
            vec![("x".to_string(), QfLiaType::Int)],
            QfLiaType::Int,
        )
    }

    fn opts_with_sig() -> FormatOptions {
        FormatOptions {
            signatures: vec![sample_sig()],
            ..Default::default()
        }
    }

    // -- ContractFormat -------------------------------------------------------

    #[test]
    fn test_format_name_and_extension() {
        assert_eq!(ContractFormat::Jml.name(), "JML");
        assert_eq!(ContractFormat::RustContracts.file_extension(), "rs");
        assert_eq!(ContractFormat::Json.file_extension(), "json");
    }

    // -- JmlFormatter ---------------------------------------------------------

    #[test]
    fn test_jml_block_contains_requires_ensures() {
        let c = sample_contract();
        let opts = opts_with_sig();
        let out = JmlFormatter.format_contract(&c, &opts);
        assert!(out.contains("/*@"), "should open a JML block");
        assert!(out.contains("@*/"), "should close a JML block");
        assert!(out.contains("@ requires"), "should contain requires");
        assert!(out.contains("@ ensures"), "should contain ensures");
    }

    #[test]
    fn test_jml_includes_assignable_and_signals() {
        let c = sample_contract();
        let opts = opts_with_sig();
        let out = JmlFormatter.format_contract(&c, &opts);
        assert!(out.contains("assignable \\nothing"));
        assert!(out.contains("signals"));
    }

    #[test]
    fn test_jml_spec_public_pure() {
        let c = sample_contract();
        let opts = opts_with_sig();
        let out = JmlFormatter.format_contract(&c, &opts);
        assert!(out.contains("spec_public"));
        assert!(out.contains("pure"));
    }

    #[test]
    fn test_jml_no_pure_for_void() {
        let mut opts = opts_with_sig();
        opts.signatures = vec![FunctionSignature::new(
            "compute",
            vec![("x".to_string(), QfLiaType::Int)],
            QfLiaType::Void,
        )];
        let c = sample_contract();
        let out = JmlFormatter.format_contract(&c, &opts);
        assert!(out.contains("spec_public"));
        assert!(!out.contains("pure"));
    }

    // -- RustContractsFormatter -----------------------------------------------

    #[test]
    fn test_rust_contracts_attributes() {
        let c = sample_contract();
        let opts = opts_with_sig();
        let out = RustContractsFormatter.format_contract(&c, &opts);
        assert!(out.contains("#[requires("));
        assert!(out.contains("#[ensures(ret ->"));
    }

    #[test]
    fn test_rust_contracts_signature() {
        let c = sample_contract();
        let opts = opts_with_sig();
        let out = RustContractsFormatter.format_contract(&c, &opts);
        assert!(out.contains("fn compute(x: i32) -> i32"));
    }

    // -- JsonContractFormatter ------------------------------------------------

    #[test]
    fn test_json_roundtrip_entry() {
        let c = sample_contract();
        let opts = opts_with_sig();
        let json = JsonContractFormatter.format_contract(&c, &opts);
        let entry: JsonContractEntry = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(entry.function_name, "compute");
        assert_eq!(entry.clauses.len(), 2);
        assert_eq!(entry.clauses[0].kind, "requires");
    }

    #[test]
    fn test_json_spec_output() {
        let mut spec = Specification::new().with_program_name("test");
        spec.add_contract(sample_contract());
        let opts = opts_with_sig();
        let json = JsonContractFormatter.format_specification(&spec, &opts);
        let output: JsonContractOutput = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(output.schema_version, "1.0.0");
        assert_eq!(output.metadata.tool, "MutSpec");
        assert_eq!(output.contracts.len(), 1);
    }

    // -- SarifFormatter -------------------------------------------------------

    #[test]
    fn test_sarif_rules() {
        let rules = sarif_contract_rules();
        assert_eq!(rules.len(), 4);
        assert!(rules.iter().any(|r| r.id == "mutspec/requires-violation"));
    }

    #[test]
    fn test_sarif_results() {
        let c = sample_contract();
        let opts = FormatOptions::default();
        let out = SarifFormatter.format_contract(&c, &opts);
        let results: Vec<SarifContractResult> =
            serde_json::from_str(&out).expect("valid JSON array");
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.clause_kind == "requires"));
    }

    #[test]
    fn test_sarif_spec_output() {
        let mut spec = Specification::new();
        spec.add_contract(sample_contract());
        let opts = FormatOptions::default();
        let json = SarifFormatter.format_specification(&spec, &opts);
        let output: SarifOutput = serde_json::from_str(&json).expect("valid SARIF JSON");
        assert_eq!(output.version, "2.1.0");
        assert!(!output.results.is_empty());
    }

    // -- TextFormatter --------------------------------------------------------

    #[test]
    fn test_text_formatter() {
        let c = sample_contract();
        let opts = FormatOptions::default();
        let out = TextFormatter.format_contract(&c, &opts);
        assert!(out.contains("compute"));
        assert!(out.contains("requires"));
    }

    // -- formatter_for --------------------------------------------------------

    #[test]
    fn test_formatter_for_dispatches() {
        let f = formatter_for(ContractFormat::Text);
        let c = sample_contract();
        let opts = FormatOptions::default();
        let out = f.format_contract(&c, &opts);
        assert!(out.contains("compute"));
    }
}

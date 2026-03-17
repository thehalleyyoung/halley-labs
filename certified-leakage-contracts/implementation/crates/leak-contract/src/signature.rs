//! Human-readable contract signatures.
//!
//! A contract signature is a compact, deterministic textual representation
//! of a leakage contract that can be used for diffing, display, and hashing.
//!
//! Example:
//! ```text
//! aes_encrypt(pre: {alignment, assoc≥8}) → [τ: 4 sets, B: 2.00 bits, exact]
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::contract::{ContractStrength, LeakageContract};
use shared_types::FunctionId;

// ---------------------------------------------------------------------------
// Function signature
// ---------------------------------------------------------------------------

/// Describes the input/output shape of a function independent of its
/// leakage contract (analogous to a type signature).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionSignature {
    /// Fully qualified function name.
    pub name: String,
    /// Function identifier in the binary.
    pub function_id: FunctionId,
    /// Number of input parameters.
    pub num_inputs: usize,
    /// Number of output parameters.
    pub num_outputs: usize,
    /// Whether the function is a leaf (makes no further calls).
    pub is_leaf: bool,
    /// Calling convention (e.g., `"sysv64"`).
    pub calling_convention: String,
}

impl FunctionSignature {
    /// Create a new function signature.
    pub fn new(name: impl Into<String>, function_id: FunctionId) -> Self {
        Self {
            name: name.into(),
            function_id,
            num_inputs: 0,
            num_outputs: 0,
            is_leaf: true,
            calling_convention: "sysv64".into(),
        }
    }

    /// Builder: set number of inputs.
    pub fn with_inputs(mut self, n: usize) -> Self {
        self.num_inputs = n;
        self
    }

    /// Builder: set number of outputs.
    pub fn with_outputs(mut self, n: usize) -> Self {
        self.num_outputs = n;
        self
    }

    /// Builder: mark as non-leaf.
    pub fn non_leaf(mut self) -> Self {
        self.is_leaf = false;
        self
    }
}

impl fmt::Display for FunctionSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({} in, {} out{})",
            self.name,
            self.num_inputs,
            self.num_outputs,
            if self.is_leaf { ", leaf" } else { "" }
        )
    }
}

// ---------------------------------------------------------------------------
// Contract signature
// ---------------------------------------------------------------------------

/// A compact textual representation of a leakage contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContractSignature {
    /// The function signature portion.
    pub function_sig: FunctionSignature,
    /// Compact string encoding of the cache transformer.
    pub transformer_summary: String,
    /// Leakage bound in bits (formatted).
    pub bound_summary: String,
    /// Contract strength label.
    pub strength_label: String,
    /// Canonical signature string (computed lazily).
    pub canonical: String,
}

impl ContractSignature {
    /// Build a signature from a full contract.
    pub fn from_contract(contract: &LeakageContract) -> Self {
        let func_sig = FunctionSignature::new(
            &contract.function_name,
            contract.function_id.clone(),
        );
        let transformer_summary = format!(
            "{} sets modified",
            contract.cache_transformer.modified_sets()
        );
        let bound_summary = format!("{:.2} bits", contract.worst_case_bits());
        let strength_label = contract.strength.label().to_string();

        let canonical = format!(
            "{}(…) → [τ: {}, B: {}, {}]",
            contract.function_name, transformer_summary, bound_summary, strength_label
        );

        Self {
            function_sig: func_sig,
            transformer_summary,
            bound_summary,
            strength_label,
            canonical,
        }
    }

    /// Returns the canonical signature string.
    pub fn as_str(&self) -> &str {
        &self.canonical
    }
}

impl fmt::Display for ContractSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.canonical)
    }
}

// ---------------------------------------------------------------------------
// Signature parser
// ---------------------------------------------------------------------------

/// Parses a canonical contract signature string back into a
/// [`ContractSignature`].
#[derive(Debug, Clone)]
pub struct SignatureParser {
    /// Whether to accept partial / incomplete signatures.
    pub lenient: bool,
}

impl SignatureParser {
    /// Create a strict parser.
    pub fn new() -> Self {
        Self { lenient: false }
    }

    /// Create a lenient parser.
    pub fn lenient() -> Self {
        Self { lenient: true }
    }

    /// Parse a canonical signature string.
    ///
    /// Returns `None` if the string does not match the expected format.
    pub fn parse(&self, input: &str) -> Option<ContractSignature> {
        // Expect: "name(…) → [τ: ..., B: ..., strength]"
        let arrow_pos = input.find('→')?;
        let name_end = input.find('(')?;
        let name = input[..name_end].trim().to_string();

        let bracket_start = input.find('[')?;
        let bracket_end = input.rfind(']')?;
        let inner = &input[bracket_start + 1..bracket_end];

        let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
        if parts.len() < 3 && !self.lenient {
            return None;
        }

        let transformer_summary = parts
            .first()
            .and_then(|p| p.strip_prefix("τ:").or_else(|| p.strip_prefix("τ: ")))
            .unwrap_or("")
            .trim()
            .to_string();

        let bound_summary = parts
            .get(1)
            .and_then(|p| p.strip_prefix("B:").or_else(|| p.strip_prefix("B: ")))
            .unwrap_or("")
            .trim()
            .to_string();

        let strength_label = parts.get(2).unwrap_or(&"").trim().to_string();

        let func_sig = FunctionSignature::new(
            &name,
            FunctionId::new(0),
        );

        Some(ContractSignature {
            function_sig: func_sig,
            transformer_summary,
            bound_summary,
            strength_label,
            canonical: input.to_string(),
        })
    }
}

impl Default for SignatureParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Signature formatter
// ---------------------------------------------------------------------------

/// Formats a [`ContractSignature`] in various styles.
#[derive(Debug, Clone)]
pub struct SignatureFormatter {
    /// Whether to include the full transformer summary.
    pub include_transformer: bool,
    /// Whether to include the strength label.
    pub include_strength: bool,
    /// Maximum width for the output line.
    pub max_width: Option<usize>,
}

impl SignatureFormatter {
    /// Create a default formatter (full detail).
    pub fn new() -> Self {
        Self {
            include_transformer: true,
            include_strength: true,
            max_width: None,
        }
    }

    /// Create a compact formatter.
    pub fn compact() -> Self {
        Self {
            include_transformer: false,
            include_strength: false,
            max_width: Some(60),
        }
    }

    /// Format a signature.
    pub fn format(&self, sig: &ContractSignature) -> String {
        let mut parts = Vec::new();
        if self.include_transformer {
            parts.push(format!("τ: {}", sig.transformer_summary));
        }
        parts.push(format!("B: {}", sig.bound_summary));
        if self.include_strength {
            parts.push(sig.strength_label.clone());
        }

        let result = format!(
            "{}(…) → [{}]",
            sig.function_sig.name,
            parts.join(", ")
        );

        if let Some(max) = self.max_width {
            if result.len() > max {
                return format!("{}…", &result[..max - 1]);
            }
        }

        result
    }
}

impl Default for SignatureFormatter {
    fn default() -> Self {
        Self::new()
    }
}

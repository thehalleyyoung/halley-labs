//! Convert PIT mutation descriptors to MutSpec internal types.
//!
//! PIT uses its own naming conventions for mutators, method descriptors, and
//! detection statuses.  This module bridges the gap by mapping:
//!
//! * PIT mutator class names → [`MutationOperator`]
//! * PIT detection statuses → [`MutantStatus`]
//! * [`PitMutation`] → [`MutantDescriptor`] (with an associated [`MutantId`])
//! * JVM method descriptors → human-readable signatures
//!
//! All conversions are deterministic and reproducible: the same PIT mutation
//! always maps to the same MutSpec representation.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use shared_types::{
    KillInfo, MutantDescriptor, MutantId, MutantStatus, MutationOperator, MutationSite,
    SourceLocation, SpanInfo,
};
use uuid::Uuid;

use crate::errors::{PitError, PitResult};
use crate::parser::{PitDetectionStatus, PitMutation};

// ---------------------------------------------------------------------------
// PIT mutator class → MutSpec operator mapping
// ---------------------------------------------------------------------------

/// Mapping table from PIT mutator *short* class names to MutSpec operators.
///
/// PIT ships with the "Gregor" mutation engine whose mutators live under
/// `org.pitest.mutationtest.engine.gregor.mutators.*`.  We match on the
/// unqualified class name so that custom PIT plugin mutators with the same
/// short name are handled identically.
static PIT_MUTATOR_MAP: &[(&str, MutationOperator)] = &[
    // --- Math mutations → AOR ---
    ("MathMutator", MutationOperator::Aor),
    ("MATH", MutationOperator::Aor),
    ("MathMutator_MATH", MutationOperator::Aor),
    // --- Negate conditionals → ROR ---
    ("NegateConditionalsMutator", MutationOperator::Ror),
    ("NEGATE_CONDITIONALS", MutationOperator::Ror),
    (
        "NegateConditionalsMutator_NEGATE_CONDITIONALS",
        MutationOperator::Ror,
    ),
    // --- Conditionals boundary → ROR ---
    ("ConditionalsBoundaryMutator", MutationOperator::Ror),
    ("CONDITIONALS_BOUNDARY", MutationOperator::Ror),
    (
        "ConditionalsBoundaryMutator_CONDITIONALS_BOUNDARY",
        MutationOperator::Ror,
    ),
    // --- Increments → AOR (increment/decrement changes) ---
    ("IncrementsMutator", MutationOperator::Aor),
    ("INCREMENTS", MutationOperator::Aor),
    ("IncrementsMutator_INCREMENTS", MutationOperator::Aor),
    // --- Invert negatives → UOI ---
    ("InvertNegsMutator", MutationOperator::Uoi),
    ("INVERT_NEGS", MutationOperator::Uoi),
    ("InvertNegsMutator_INVERT_NEGS", MutationOperator::Uoi),
    // --- Return values → RVR ---
    ("ReturnValsMutator", MutationOperator::Rvr),
    ("RETURN_VALS", MutationOperator::Rvr),
    ("BooleanTrueReturnValsMutator", MutationOperator::Rvr),
    ("TRUE_RETURNS", MutationOperator::Rvr),
    ("BooleanFalseReturnValsMutator", MutationOperator::Rvr),
    ("FALSE_RETURNS", MutationOperator::Rvr),
    ("PrimitiveReturnsMutator", MutationOperator::Rvr),
    ("PRIMITIVE_RETURNS", MutationOperator::Rvr),
    ("EmptyObjectReturnValsMutator", MutationOperator::Rvr),
    ("EMPTY_RETURNS", MutationOperator::Rvr),
    ("NullReturnValsMutator", MutationOperator::Rvr),
    ("NULL_RETURNS", MutationOperator::Rvr),
    // --- Void method calls → SDL ---
    ("VoidMethodCallMutator", MutationOperator::Sdl),
    ("VOID_METHOD_CALLS", MutationOperator::Sdl),
    ("NonVoidMethodCallMutator", MutationOperator::Sdl),
    ("NON_VOID_METHOD_CALLS", MutationOperator::Sdl),
    ("RemoveConditionalMutator", MutationOperator::Sdl),
    ("REMOVE_CONDITIONALS", MutationOperator::Sdl),
    // --- Constructor calls → SDL ---
    ("ConstructorCallMutator", MutationOperator::Sdl),
    ("CONSTRUCTOR_CALLS", MutationOperator::Sdl),
    // --- Experimental: switch → COR ---
    ("SwitchMutator", MutationOperator::Cor),
    ("EXPERIMENTAL_SWITCH", MutationOperator::Cor),
    // --- Experimental: argument propagation → OSW ---
    ("ArgumentPropagationMutator", MutationOperator::Osw),
    ("EXPERIMENTAL_ARGUMENT_PROPAGATION", MutationOperator::Osw),
    // --- Experimental: naked receiver → SDL ---
    ("NakedReceiverMutator", MutationOperator::Sdl),
    ("EXPERIMENTAL_NAKED_RECEIVER", MutationOperator::Sdl),
    // --- Experimental: member variable → CRC ---
    ("MemberVariableMutator", MutationOperator::Crc),
    ("EXPERIMENTAL_MEMBER_VARIABLE", MutationOperator::Crc),
    // --- Experimental: big integer → AOR ---
    ("BigIntegerMutator", MutationOperator::Aor),
    ("EXPERIMENTAL_BIG_INTEGER", MutationOperator::Aor),
];

/// Look up the MutSpec [`MutationOperator`] for a PIT mutator string.
///
/// The input can be a fully-qualified class name, a short class name, or one
/// of PIT's enum-style identifiers (e.g. `MATH`, `NEGATE_CONDITIONALS`).
///
/// Returns `None` if no mapping exists.
pub fn pit_mutator_to_operator(mutator: &str) -> Option<MutationOperator> {
    // Try short name first (last segment).
    let short = mutator.rsplit('.').next().unwrap_or(mutator);
    for &(name, op) in PIT_MUTATOR_MAP {
        if name.eq_ignore_ascii_case(short) {
            return Some(op);
        }
    }
    // Also try the full string in case it matches exactly.
    for &(name, op) in PIT_MUTATOR_MAP {
        if name.eq_ignore_ascii_case(mutator) {
            return Some(op);
        }
    }
    None
}

/// Return all known PIT mutator name → MutSpec operator pairs.
pub fn all_mutator_mappings() -> &'static [(&'static str, MutationOperator)] {
    PIT_MUTATOR_MAP
}

// ---------------------------------------------------------------------------
// JVM method descriptor parsing
// ---------------------------------------------------------------------------

/// A parsed JVM method descriptor.
///
/// JVM method descriptors have the form `(ParameterTypes)ReturnType` where
/// each type is encoded as a single character or a class reference
/// (e.g. `(ILjava/lang/String;)V`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JvmMethodDescriptor {
    /// The raw descriptor string.
    pub raw: String,
    /// Parsed parameter type names.
    pub parameters: Vec<String>,
    /// Parsed return type name.
    pub return_type: String,
}

use serde::{Deserialize, Serialize};

impl JvmMethodDescriptor {
    /// Parse a JVM method descriptor string.
    ///
    /// # Examples
    /// ```ignore
    /// let desc = JvmMethodDescriptor::parse("(II)I").unwrap();
    /// assert_eq!(desc.parameters, vec!["int", "int"]);
    /// assert_eq!(desc.return_type, "int");
    /// ```
    pub fn parse(descriptor: &str) -> PitResult<Self> {
        if !descriptor.starts_with('(') {
            return Err(PitError::InvalidDescriptor {
                descriptor: descriptor.to_string(),
                reason: "must start with '('".into(),
            });
        }
        let close = descriptor
            .find(')')
            .ok_or_else(|| PitError::InvalidDescriptor {
                descriptor: descriptor.to_string(),
                reason: "missing ')'".into(),
            })?;

        let params_str = &descriptor[1..close];
        let return_str = &descriptor[close + 1..];

        let parameters = Self::parse_types(params_str, descriptor)?;
        let mut ret_types = Self::parse_types(return_str, descriptor)?;
        let return_type = if ret_types.len() == 1 {
            ret_types.remove(0)
        } else if return_str == "V" {
            "void".to_string()
        } else {
            return Err(PitError::InvalidDescriptor {
                descriptor: descriptor.to_string(),
                reason: format!("invalid return type: `{return_str}`"),
            });
        };

        Ok(JvmMethodDescriptor {
            raw: descriptor.to_string(),
            parameters,
            return_type,
        })
    }

    /// Parse a sequence of JVM type encodings.
    fn parse_types(s: &str, full_desc: &str) -> PitResult<Vec<String>> {
        let mut types = Vec::new();
        let bytes = s.as_bytes();
        let mut i = 0;

        while i < bytes.len() {
            match bytes[i] {
                b'B' => {
                    types.push("byte".into());
                    i += 1;
                }
                b'C' => {
                    types.push("char".into());
                    i += 1;
                }
                b'D' => {
                    types.push("double".into());
                    i += 1;
                }
                b'F' => {
                    types.push("float".into());
                    i += 1;
                }
                b'I' => {
                    types.push("int".into());
                    i += 1;
                }
                b'J' => {
                    types.push("long".into());
                    i += 1;
                }
                b'S' => {
                    types.push("short".into());
                    i += 1;
                }
                b'Z' => {
                    types.push("boolean".into());
                    i += 1;
                }
                b'V' => {
                    types.push("void".into());
                    i += 1;
                }
                b'[' => {
                    // Array: count dimensions, then parse element type.
                    let mut dims = 0;
                    while i < bytes.len() && bytes[i] == b'[' {
                        dims += 1;
                        i += 1;
                    }
                    // Parse the element type.
                    let rest = &s[i..];
                    let mut elem = Self::parse_types(
                        &rest[..Self::single_type_len(rest, full_desc)?],
                        full_desc,
                    )?;
                    if elem.len() != 1 {
                        return Err(PitError::InvalidDescriptor {
                            descriptor: full_desc.to_string(),
                            reason: "array element parse error".into(),
                        });
                    }
                    let mut name = elem.remove(0);
                    for _ in 0..dims {
                        name.push_str("[]");
                    }
                    i += Self::single_type_len(rest, full_desc)?;
                    types.push(name);
                }
                b'L' => {
                    // Object reference: L<classname>;
                    let semi = s[i..]
                        .find(';')
                        .ok_or_else(|| PitError::InvalidDescriptor {
                            descriptor: full_desc.to_string(),
                            reason: "unterminated object type".into(),
                        })?;
                    let class_name = &s[i + 1..i + semi];
                    types.push(class_name.replace('/', "."));
                    i += semi + 1;
                }
                _ => {
                    return Err(PitError::InvalidDescriptor {
                        descriptor: full_desc.to_string(),
                        reason: format!("unexpected type char `{}`", bytes[i] as char),
                    });
                }
            }
        }
        Ok(types)
    }

    /// Return the byte length of a single type encoding starting at `s[0]`.
    fn single_type_len(s: &str, full_desc: &str) -> PitResult<usize> {
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            return Err(PitError::InvalidDescriptor {
                descriptor: full_desc.to_string(),
                reason: "empty type".into(),
            });
        }
        match bytes[0] {
            b'B' | b'C' | b'D' | b'F' | b'I' | b'J' | b'S' | b'Z' | b'V' => Ok(1),
            b'L' => {
                let semi = s.find(';').ok_or_else(|| PitError::InvalidDescriptor {
                    descriptor: full_desc.to_string(),
                    reason: "unterminated object type".into(),
                })?;
                Ok(semi + 1)
            }
            b'[' => {
                let inner_start = 1;
                Ok(inner_start + Self::single_type_len(&s[inner_start..], full_desc)?)
            }
            c => Err(PitError::InvalidDescriptor {
                descriptor: full_desc.to_string(),
                reason: format!("unexpected char `{}`", c as char),
            }),
        }
    }

    /// Render as a human-readable signature fragment: `(int, int) -> int`.
    pub fn to_readable(&self) -> String {
        let params = self.parameters.join(", ");
        format!("({params}) -> {}", self.return_type)
    }
}

impl fmt::Display for JvmMethodDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_readable())
    }
}

// ---------------------------------------------------------------------------
// Status conversion
// ---------------------------------------------------------------------------

/// Convert a [`PitDetectionStatus`] to a MutSpec [`MutantStatus`].
pub fn pit_status_to_mutant_status(status: &PitDetectionStatus) -> MutantStatus {
    match status {
        PitDetectionStatus::Killed => MutantStatus::Killed,
        PitDetectionStatus::Survived => MutantStatus::Alive,
        PitDetectionStatus::TimedOut => MutantStatus::Timeout,
        PitDetectionStatus::NoCoverage => MutantStatus::Alive,
        PitDetectionStatus::NonViable => MutantStatus::Error("non-viable mutant".into()),
        PitDetectionStatus::MemoryError => MutantStatus::Error("memory error".into()),
        PitDetectionStatus::RunError => MutantStatus::Error("run error".into()),
        PitDetectionStatus::NotStarted => MutantStatus::Error("not started".into()),
        PitDetectionStatus::Started => MutantStatus::Error("started but no result".into()),
    }
}

/// Convert a MutSpec [`MutantStatus`] back to the nearest [`PitDetectionStatus`].
pub fn mutant_status_to_pit(status: &MutantStatus) -> PitDetectionStatus {
    match status {
        MutantStatus::Killed => PitDetectionStatus::Killed,
        MutantStatus::Alive => PitDetectionStatus::Survived,
        MutantStatus::Timeout => PitDetectionStatus::TimedOut,
        MutantStatus::Equivalent => PitDetectionStatus::Survived,
        MutantStatus::Error(_) => PitDetectionStatus::RunError,
    }
}

// ---------------------------------------------------------------------------
// PitMutation → MutantDescriptor conversion
// ---------------------------------------------------------------------------

/// Options controlling how PIT mutations are converted to MutSpec descriptors.
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// If true, generate deterministic UUIDs from the mutation's canonical key
    /// (reproducible across runs).  If false, use random UUIDs.
    pub deterministic_ids: bool,
    /// Base path to prepend to source file paths for [`SpanInfo`].
    pub source_root: Option<PathBuf>,
    /// If true, skip mutations whose mutator cannot be mapped to a MutSpec operator
    /// rather than returning an error.
    pub skip_unknown_mutators: bool,
    /// Default execution time (ms) to record for kill info when PIT does not
    /// report timing.
    pub default_execution_time_ms: f64,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            deterministic_ids: true,
            source_root: None,
            skip_unknown_mutators: true,
            default_execution_time_ms: 0.0,
        }
    }
}

impl ConversionOptions {
    /// Create options with deterministic IDs enabled.
    pub fn deterministic() -> Self {
        Self {
            deterministic_ids: true,
            ..Default::default()
        }
    }

    /// Create options with random IDs.
    pub fn random_ids() -> Self {
        Self {
            deterministic_ids: false,
            ..Default::default()
        }
    }

    /// Set the source root directory.
    pub fn with_source_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.source_root = Some(root.into());
        self
    }

    /// Set whether to skip unknown mutators.
    pub fn with_skip_unknown(mut self, skip: bool) -> Self {
        self.skip_unknown_mutators = skip;
        self
    }
}

/// Convert a single [`PitMutation`] to a MutSpec [`MutantDescriptor`].
///
/// Returns `Ok(None)` when the mutator is unknown and `opts.skip_unknown_mutators`
/// is true.  Returns `Err` when the mutator is unknown and skipping is disabled.
pub fn convert_pit_mutation(
    pit: &PitMutation,
    opts: &ConversionOptions,
) -> PitResult<Option<MutantDescriptor>> {
    let operator = match pit_mutator_to_operator(&pit.mutator) {
        Some(op) => op,
        None if opts.skip_unknown_mutators => {
            log::debug!("Skipping unknown mutator: {}", pit.mutator);
            return Ok(None);
        }
        None => {
            return Err(PitError::UnknownMutator {
                path: PathBuf::new(),
                mutator_class: pit.mutator.clone(),
            });
        }
    };

    let mutant_id = if opts.deterministic_ids {
        deterministic_mutant_id(pit)
    } else {
        MutantId::new()
    };

    let source_path = if let Some(ref root) = opts.source_root {
        root.join(&pit.source_file)
    } else {
        PathBuf::from(&pit.source_file)
    };

    let location = SpanInfo::point(SourceLocation::new(
        source_path,
        pit.line_number as usize,
        0,
    ));

    let original = pit
        .description
        .as_deref()
        .unwrap_or("(original)")
        .to_string();
    let replacement = format!(
        "[{} mutation by {}]",
        operator.mnemonic(),
        pit.short_mutator_name()
    );

    let site = MutationSite::new(location, operator, original, replacement)
        .with_function(&pit.mutated_method);

    let mut descriptor = MutantDescriptor::new(operator, site).with_id(mutant_id);

    // Apply status.
    let mutspec_status = pit_status_to_mutant_status(&pit.status);
    match mutspec_status {
        MutantStatus::Killed => {
            let test_name = pit
                .killing_test
                .clone()
                .unwrap_or_else(|| "(unknown)".into());
            let kill_info = KillInfo::new(test_name, opts.default_execution_time_ms);
            descriptor.mark_killed(kill_info);
        }
        MutantStatus::Timeout => {
            descriptor.mark_timeout();
        }
        MutantStatus::Error(ref msg) => {
            descriptor.mark_error(msg.clone());
        }
        MutantStatus::Equivalent => {
            descriptor.mark_equivalent("PIT equivalent detection");
        }
        MutantStatus::Alive => { /* default status */ }
    }

    Ok(Some(descriptor))
}

/// Convert a batch of PIT mutations, collecting results and skipped entries.
pub fn convert_pit_mutations(
    mutations: &[PitMutation],
    opts: &ConversionOptions,
) -> PitResult<ConversionResult> {
    let mut descriptors = Vec::with_capacity(mutations.len());
    let mut skipped = Vec::new();
    let mut errors = Vec::new();

    for (idx, pit) in mutations.iter().enumerate() {
        match convert_pit_mutation(pit, opts) {
            Ok(Some(desc)) => descriptors.push(desc),
            Ok(None) => skipped.push(idx),
            Err(e) => errors.push((idx, e.to_string())),
        }
    }

    log::info!(
        "Converted {}/{} PIT mutations ({} skipped, {} errors)",
        descriptors.len(),
        mutations.len(),
        skipped.len(),
        errors.len()
    );

    Ok(ConversionResult {
        descriptors,
        skipped_indices: skipped,
        error_indices: errors,
        total_input: mutations.len(),
    })
}

/// Result of a batch conversion operation.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Successfully converted MutSpec descriptors.
    pub descriptors: Vec<MutantDescriptor>,
    /// Indices of PIT mutations that were skipped (unknown mutator).
    pub skipped_indices: Vec<usize>,
    /// Indices and error messages for PIT mutations that failed conversion.
    pub error_indices: Vec<(usize, String)>,
    /// Total number of input PIT mutations.
    pub total_input: usize,
}

impl ConversionResult {
    /// Number of successfully converted mutations.
    pub fn converted_count(&self) -> usize {
        self.descriptors.len()
    }

    /// Conversion success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total_input == 0 {
            return 100.0;
        }
        (self.descriptors.len() as f64 / self.total_input as f64) * 100.0
    }

    /// Returns true if all mutations were converted (none skipped, none errored).
    pub fn is_complete(&self) -> bool {
        self.skipped_indices.is_empty() && self.error_indices.is_empty()
    }

    /// Build a map from MutantId to MutantDescriptor for fast lookup.
    pub fn descriptor_map(&self) -> HashMap<MutantId, &MutantDescriptor> {
        self.descriptors.iter().map(|d| (d.id.clone(), d)).collect()
    }
}

impl fmt::Display for ConversionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Converted {}/{} ({:.1}%)",
            self.converted_count(),
            self.total_input,
            self.success_rate()
        )?;
        if !self.skipped_indices.is_empty() {
            write!(f, ", {} skipped", self.skipped_indices.len())?;
        }
        if !self.error_indices.is_empty() {
            write!(f, ", {} errors", self.error_indices.len())?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Deterministic ID generation
// ---------------------------------------------------------------------------

/// Generate a deterministic [`MutantId`] from the mutation's canonical key.
///
/// Uses UUID v5 (SHA-1 namespace) so the same PIT mutation always produces
/// the same MutantId, enabling stable cross-run comparisons.
fn deterministic_mutant_id(pit: &PitMutation) -> MutantId {
    let key = pit.canonical_key();
    let namespace = Uuid::NAMESPACE_OID;
    let uuid = Uuid::new_v5(&namespace, key.as_bytes());
    MutantId::from_uuid(uuid)
}

// ---------------------------------------------------------------------------
// Reverse mapping helper
// ---------------------------------------------------------------------------

/// Given a MutSpec [`MutationOperator`], return the primary PIT mutator
/// class short name that corresponds to it.
pub fn operator_to_pit_mutator(op: &MutationOperator) -> &'static str {
    match op {
        MutationOperator::Aor => "MathMutator",
        MutationOperator::Ror => "NegateConditionalsMutator",
        MutationOperator::Lcr => "NegateConditionalsMutator",
        MutationOperator::Uoi => "InvertNegsMutator",
        MutationOperator::Abs => "InvertNegsMutator",
        MutationOperator::Cor => "SwitchMutator",
        MutationOperator::Sdl => "VoidMethodCallMutator",
        MutationOperator::Rvr => "ReturnValsMutator",
        MutationOperator::Crc => "MemberVariableMutator",
        MutationOperator::Air => "IncrementsMutator",
        MutationOperator::Osw => "ArgumentPropagationMutator",
        MutationOperator::Bcn => "NegateConditionalsMutator",
    }
}

/// Return all PIT mutator class names that map to the given operator.
pub fn operator_to_all_pit_mutators(op: &MutationOperator) -> Vec<&'static str> {
    PIT_MUTATOR_MAP
        .iter()
        .filter(|(_, mapped_op)| mapped_op == op)
        .map(|(name, _)| *name)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::PitDetectionStatus;

    fn sample_pit_mutation() -> PitMutation {
        PitMutation {
            detected: true,
            status: PitDetectionStatus::Killed,
            number_of_tests_run: 5,
            source_file: "Calculator.java".into(),
            mutated_class: "com.example.Calculator".into(),
            mutated_method: "add".into(),
            method_description: "(II)I".into(),
            line_number: 42,
            mutator: "org.pitest.mutationtest.engine.gregor.mutators.MathMutator".into(),
            indexes: vec![12],
            blocks: vec![3],
            killing_test: Some("com.example.CalculatorTest::testAdd".into()),
            description: Some("replaced int return with 0".into()),
        }
    }

    #[test]
    fn test_pit_mutator_to_operator_math() {
        assert_eq!(
            pit_mutator_to_operator("MathMutator"),
            Some(MutationOperator::Aor)
        );
        assert_eq!(pit_mutator_to_operator("MATH"), Some(MutationOperator::Aor));
    }

    #[test]
    fn test_pit_mutator_to_operator_fqn() {
        let fqn = "org.pitest.mutationtest.engine.gregor.mutators.MathMutator";
        assert_eq!(pit_mutator_to_operator(fqn), Some(MutationOperator::Aor));
    }

    #[test]
    fn test_pit_mutator_to_operator_negate() {
        assert_eq!(
            pit_mutator_to_operator("NegateConditionalsMutator"),
            Some(MutationOperator::Ror)
        );
        assert_eq!(
            pit_mutator_to_operator("NEGATE_CONDITIONALS"),
            Some(MutationOperator::Ror)
        );
    }

    #[test]
    fn test_pit_mutator_to_operator_return_vals() {
        assert_eq!(
            pit_mutator_to_operator("ReturnValsMutator"),
            Some(MutationOperator::Rvr)
        );
        assert_eq!(
            pit_mutator_to_operator("BooleanTrueReturnValsMutator"),
            Some(MutationOperator::Rvr)
        );
    }

    #[test]
    fn test_pit_mutator_to_operator_unknown() {
        assert_eq!(pit_mutator_to_operator("SomeCustomMutator"), None);
    }

    #[test]
    fn test_pit_mutator_case_insensitive() {
        assert_eq!(
            pit_mutator_to_operator("mathmutator"),
            Some(MutationOperator::Aor)
        );
    }

    #[test]
    fn test_jvm_descriptor_simple() {
        let desc = JvmMethodDescriptor::parse("(II)I").unwrap();
        assert_eq!(desc.parameters, vec!["int", "int"]);
        assert_eq!(desc.return_type, "int");
        assert_eq!(desc.to_readable(), "(int, int) -> int");
    }

    #[test]
    fn test_jvm_descriptor_void() {
        let desc = JvmMethodDescriptor::parse("()V").unwrap();
        assert!(desc.parameters.is_empty());
        assert_eq!(desc.return_type, "void");
    }

    #[test]
    fn test_jvm_descriptor_object_params() {
        let desc = JvmMethodDescriptor::parse("(Ljava/lang/String;I)Z").unwrap();
        assert_eq!(desc.parameters, vec!["java.lang.String", "int"]);
        assert_eq!(desc.return_type, "boolean");
    }

    #[test]
    fn test_jvm_descriptor_array() {
        let desc = JvmMethodDescriptor::parse("([I)V").unwrap();
        assert_eq!(desc.parameters, vec!["int[]"]);
        assert_eq!(desc.return_type, "void");
    }

    #[test]
    fn test_jvm_descriptor_multi_dim_array() {
        let desc = JvmMethodDescriptor::parse("([[I)V").unwrap();
        assert_eq!(desc.parameters, vec!["int[][]"]);
    }

    #[test]
    fn test_jvm_descriptor_complex() {
        let desc =
            JvmMethodDescriptor::parse("(ILjava/util/List;[Ljava/lang/String;)Ljava/lang/Object;")
                .unwrap();
        assert_eq!(desc.parameters.len(), 3);
        assert_eq!(desc.parameters[0], "int");
        assert_eq!(desc.parameters[1], "java.util.List");
        assert_eq!(desc.parameters[2], "java.lang.String[]");
        assert_eq!(desc.return_type, "java.lang.Object");
    }

    #[test]
    fn test_jvm_descriptor_invalid_no_parens() {
        assert!(JvmMethodDescriptor::parse("II").is_err());
    }

    #[test]
    fn test_jvm_descriptor_invalid_no_close() {
        assert!(JvmMethodDescriptor::parse("(II").is_err());
    }

    #[test]
    fn test_jvm_descriptor_display() {
        let desc = JvmMethodDescriptor::parse("(IJ)Z").unwrap();
        assert_eq!(desc.to_string(), "(int, long) -> boolean");
    }

    #[test]
    fn test_pit_status_to_mutant_status() {
        assert_eq!(
            pit_status_to_mutant_status(&PitDetectionStatus::Killed),
            MutantStatus::Killed
        );
        assert_eq!(
            pit_status_to_mutant_status(&PitDetectionStatus::Survived),
            MutantStatus::Alive
        );
        assert_eq!(
            pit_status_to_mutant_status(&PitDetectionStatus::TimedOut),
            MutantStatus::Timeout
        );
        assert!(pit_status_to_mutant_status(&PitDetectionStatus::NonViable).is_error());
    }

    #[test]
    fn test_mutant_status_to_pit() {
        assert_eq!(
            mutant_status_to_pit(&MutantStatus::Killed),
            PitDetectionStatus::Killed
        );
        assert_eq!(
            mutant_status_to_pit(&MutantStatus::Alive),
            PitDetectionStatus::Survived
        );
        assert_eq!(
            mutant_status_to_pit(&MutantStatus::Timeout),
            PitDetectionStatus::TimedOut
        );
    }

    #[test]
    fn test_convert_pit_mutation_killed() {
        let pit = sample_pit_mutation();
        let opts = ConversionOptions::deterministic();
        let result = convert_pit_mutation(&pit, &opts).unwrap().unwrap();

        assert_eq!(result.operator, MutationOperator::Aor);
        assert!(result.is_killed());
        assert!(result.kill_info.is_some());
        assert!(result
            .kill_info
            .as_ref()
            .unwrap()
            .test_name
            .contains("testAdd"));
    }

    #[test]
    fn test_convert_pit_mutation_survived() {
        let mut pit = sample_pit_mutation();
        pit.status = PitDetectionStatus::Survived;
        pit.detected = false;
        pit.killing_test = None;

        let opts = ConversionOptions::deterministic();
        let result = convert_pit_mutation(&pit, &opts).unwrap().unwrap();
        assert!(result.is_alive());
    }

    #[test]
    fn test_convert_pit_mutation_unknown_skip() {
        let mut pit = sample_pit_mutation();
        pit.mutator = "com.custom.UnknownMutator".into();

        let opts = ConversionOptions::default();
        let result = convert_pit_mutation(&pit, &opts).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_convert_pit_mutation_unknown_error() {
        let mut pit = sample_pit_mutation();
        pit.mutator = "com.custom.UnknownMutator".into();

        let opts = ConversionOptions::default().with_skip_unknown(false);
        assert!(convert_pit_mutation(&pit, &opts).is_err());
    }

    #[test]
    fn test_deterministic_ids() {
        let pit = sample_pit_mutation();
        let id1 = deterministic_mutant_id(&pit);
        let id2 = deterministic_mutant_id(&pit);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_convert_batch() {
        let mut killed = sample_pit_mutation();
        let mut survived = sample_pit_mutation();
        survived.status = PitDetectionStatus::Survived;
        survived.detected = false;
        survived.killing_test = None;
        survived.mutated_method = "subtract".into();
        survived.line_number = 56;

        let mutations = vec![killed, survived];
        let opts = ConversionOptions::deterministic();
        let result = convert_pit_mutations(&mutations, &opts).unwrap();

        assert_eq!(result.converted_count(), 2);
        assert!(result.is_complete());
        assert!(result.success_rate() > 99.0);
    }

    #[test]
    fn test_conversion_result_display() {
        let result = ConversionResult {
            descriptors: Vec::new(),
            skipped_indices: vec![0, 1],
            error_indices: vec![],
            total_input: 5,
        };
        let s = result.to_string();
        assert!(s.contains("0/5"));
        assert!(s.contains("2 skipped"));
    }

    #[test]
    fn test_operator_to_pit_mutator() {
        assert_eq!(
            operator_to_pit_mutator(&MutationOperator::Aor),
            "MathMutator"
        );
        assert_eq!(
            operator_to_pit_mutator(&MutationOperator::Ror),
            "NegateConditionalsMutator"
        );
    }

    #[test]
    fn test_operator_to_all_pit_mutators() {
        let aor_mutators = operator_to_all_pit_mutators(&MutationOperator::Aor);
        assert!(aor_mutators.contains(&"MathMutator"));
        assert!(aor_mutators.contains(&"IncrementsMutator"));
    }

    #[test]
    fn test_conversion_options_builder() {
        let opts = ConversionOptions::deterministic()
            .with_source_root("/src/main/java")
            .with_skip_unknown(false);
        assert!(opts.deterministic_ids);
        assert_eq!(opts.source_root, Some(PathBuf::from("/src/main/java")));
        assert!(!opts.skip_unknown_mutators);
    }

    #[test]
    fn test_all_mutator_mappings() {
        let mappings = all_mutator_mappings();
        assert!(mappings.len() > 20);
        assert!(mappings.iter().any(|(name, _)| *name == "MathMutator"));
    }

    #[test]
    fn test_jvm_descriptor_all_primitives() {
        let desc = JvmMethodDescriptor::parse("(BCDFIJSZ)V").unwrap();
        assert_eq!(desc.parameters.len(), 8);
        assert_eq!(desc.parameters[0], "byte");
        assert_eq!(desc.parameters[1], "char");
        assert_eq!(desc.parameters[2], "double");
        assert_eq!(desc.parameters[3], "float");
        assert_eq!(desc.parameters[4], "int");
        assert_eq!(desc.parameters[5], "long");
        assert_eq!(desc.parameters[6], "short");
        assert_eq!(desc.parameters[7], "boolean");
    }

    #[test]
    fn test_descriptor_map() {
        let pit = sample_pit_mutation();
        let opts = ConversionOptions::deterministic();
        let result = convert_pit_mutations(&[pit], &opts).unwrap();
        let map = result.descriptor_map();
        assert_eq!(map.len(), 1);
    }
}

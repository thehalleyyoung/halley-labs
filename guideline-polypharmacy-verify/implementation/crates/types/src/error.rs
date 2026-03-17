//! Error types for all GuardPharma subsystems.
//!
//! Provides a unified error hierarchy with specific variants for each
//! component of the verification pipeline: parsing, pharmacokinetic modeling,
//! verification, contracts, encoding, conflict detection, recommendations,
//! clinical evaluation, and configuration.

use thiserror::Error;

/// Convenience result type alias for GuardPharma operations.
pub type Result<T> = std::result::Result<T, GuardPharmaError>;

/// Top-level error type encompassing all GuardPharma subsystem errors.
#[derive(Debug, Error, Clone)]
pub enum GuardPharmaError {
    /// Error during parsing of drug data, guidelines, or input files.
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    /// Error in pharmacokinetic model computation.
    #[error("PK model error: {0}")]
    PkModel(#[from] PkModelError),

    /// Error during formal verification (Tier 1 or Tier 2).
    #[error("verification error: {0}")]
    Verification(#[from] VerificationError),

    /// Error in contract checking or composition.
    #[error("contract error: {0}")]
    Contract(#[from] ContractError),

    /// Error in SMT/SAT encoding.
    #[error("encoding error: {0}")]
    Encoding(#[from] EncodingError),

    /// Error in conflict detection.
    #[error("conflict error: {0}")]
    Conflict(#[from] ConflictError),

    /// Error in recommendation generation.
    #[error("recommendation error: {0}")]
    Recommendation(#[from] RecommendationError),

    /// Error in clinical data processing.
    #[error("clinical error: {0}")]
    Clinical(#[from] ClinicalError),

    /// Error in configuration loading or validation.
    #[error("config error: {0}")]
    Config(#[from] ConfigError),

    /// Generic internal error.
    #[error("internal error: {message}")]
    Internal {
        /// Human-readable description of the internal error.
        message: String,
        /// Optional source context (file, function, etc.).
        context: Option<String>,
    },

    /// I/O error wrapper.
    #[error("I/O error: {message}")]
    Io {
        /// Description of the I/O failure.
        message: String,
    },

    /// Serialization / deserialization error.
    #[error("serialization error: {message}")]
    Serialization {
        /// Description of the serialization failure.
        message: String,
    },

    /// Timeout during verification or computation.
    #[error("timeout after {duration_secs}s: {operation}")]
    Timeout {
        /// The operation that timed out.
        operation: String,
        /// How many seconds elapsed before timeout.
        duration_secs: f64,
    },
}

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

/// Errors arising from parsing input data (drug databases, guidelines, etc.).
#[derive(Debug, Error, Clone)]
pub enum ParseError {
    /// Invalid drug identifier string.
    #[error("invalid drug identifier: {0}")]
    InvalidDrugId(String),

    /// Unknown drug class name.
    #[error("unknown drug class: {0}")]
    UnknownDrugClass(String),

    /// Unknown drug route.
    #[error("unknown drug route: {0}")]
    UnknownDrugRoute(String),

    /// Malformed dosing schedule.
    #[error("invalid dosing schedule: {reason}")]
    InvalidDosingSchedule {
        /// Explanation of why the schedule is invalid.
        reason: String,
    },

    /// Malformed guideline specification.
    #[error("invalid guideline at line {line}: {reason}")]
    InvalidGuideline {
        /// Line number in the source.
        line: usize,
        /// Explanation.
        reason: String,
    },

    /// Unexpected token during lexical analysis.
    #[error("unexpected token '{token}' at position {position}")]
    UnexpectedToken {
        /// The offending token text.
        token: String,
        /// Byte offset in the input.
        position: usize,
    },

    /// Missing required field in structured input.
    #[error("missing required field '{field}' in {context}")]
    MissingField {
        /// Name of the missing field.
        field: String,
        /// Context where the field was expected.
        context: String,
    },

    /// Numeric value out of acceptable range.
    #[error("value {value} out of range [{min}, {max}] for {field}")]
    ValueOutOfRange {
        /// The field name.
        field: String,
        /// The provided value.
        value: f64,
        /// Minimum acceptable value.
        min: f64,
        /// Maximum acceptable value.
        max: f64,
    },

    /// Generic parse failure.
    #[error("parse error: {0}")]
    Generic(String),
}

// ---------------------------------------------------------------------------
// PkModelError
// ---------------------------------------------------------------------------

/// Errors in pharmacokinetic model construction or evaluation.
#[derive(Debug, Error, Clone)]
pub enum PkModelError {
    /// Missing PK parameters for a drug.
    #[error("missing PK parameters for drug {drug_id}")]
    MissingParameters {
        /// The drug lacking parameters.
        drug_id: String,
    },

    /// Invalid PK parameter value.
    #[error("invalid PK parameter '{param}' = {value}: {reason}")]
    InvalidParameter {
        /// Parameter name.
        param: String,
        /// The invalid value.
        value: f64,
        /// Why it is invalid.
        reason: String,
    },

    /// Numerical instability during ODE integration.
    #[error("numerical instability at t={time}: {detail}")]
    NumericalInstability {
        /// Simulation time at which instability occurred.
        time: f64,
        /// Detail description.
        detail: String,
    },

    /// Compartment model is ill-defined.
    #[error("invalid compartment model: {0}")]
    InvalidCompartmentModel(String),

    /// Steady-state computation did not converge.
    #[error("steady-state not reached after {iterations} iterations (residual={residual})")]
    SteadyStateNotReached {
        /// Number of iterations attempted.
        iterations: usize,
        /// Residual norm.
        residual: f64,
    },

    /// Drug interaction coefficient is out of bounds.
    #[error("interaction coefficient {value} out of bounds for {drug_a} × {drug_b}")]
    InteractionCoefficientOutOfBounds {
        /// First drug.
        drug_a: String,
        /// Second drug.
        drug_b: String,
        /// The coefficient.
        value: f64,
    },
}

// ---------------------------------------------------------------------------
// VerificationError
// ---------------------------------------------------------------------------

/// Errors during formal verification runs.
#[derive(Debug, Error, Clone)]
pub enum VerificationError {
    /// Abstract interpretation did not converge within the iteration limit.
    #[error("abstract interpretation failed to converge after {iterations} iterations")]
    ConvergenceFailed {
        /// Number of iterations attempted.
        iterations: usize,
    },

    /// The property to be verified is malformed.
    #[error("malformed safety property: {0}")]
    MalformedProperty(String),

    /// State space explosion: too many reachable states.
    #[error("state space too large: {num_states} states (limit: {limit})")]
    StateSpaceExplosion {
        /// Number of states discovered.
        num_states: usize,
        /// Configured limit.
        limit: usize,
    },

    /// Tier escalation failed (Tier 1 inconclusive, Tier 2 also fails).
    #[error("tier escalation failed: tier1 result={tier1_result}, tier2 reason={tier2_reason}")]
    TierEscalationFailed {
        /// Summary of Tier 1 outcome.
        tier1_result: String,
        /// Reason Tier 2 also failed.
        tier2_reason: String,
    },

    /// PTA construction error.
    #[error("PTA construction error: {0}")]
    PtaConstruction(String),

    /// Unsupported feature in the verification backend.
    #[error("unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// Invariant violation in a PTA location.
    #[error("invariant violation at location {location}: {detail}")]
    InvariantViolation {
        /// Location where the invariant failed.
        location: String,
        /// Detail of the violation.
        detail: String,
    },
}

// ---------------------------------------------------------------------------
// ContractError
// ---------------------------------------------------------------------------

/// Errors in assume-guarantee contract operations.
#[derive(Debug, Error, Clone)]
pub enum ContractError {
    /// Two contracts have incompatible assumptions/guarantees.
    #[error("incompatible contracts {a} and {b}: {reason}")]
    Incompatible {
        /// First contract identifier.
        a: String,
        /// Second contract identifier.
        b: String,
        /// Why they are incompatible.
        reason: String,
    },

    /// A contract assumption is vacuously true (over-approximate).
    #[error("vacuous assumption in contract {contract_id}: {detail}")]
    VacuousAssumption {
        /// Contract identifier.
        contract_id: String,
        /// Detail.
        detail: String,
    },

    /// Contract composition failed.
    #[error("composition failed for contracts [{ids}]: {reason}")]
    CompositionFailed {
        /// Comma-separated contract IDs.
        ids: String,
        /// Reason for failure.
        reason: String,
    },

    /// Circular dependency among contracts.
    #[error("circular dependency detected among contracts: {cycle}")]
    CircularDependency {
        /// Textual representation of the cycle.
        cycle: String,
    },

    /// Missing contract for a required enzyme or interaction.
    #[error("missing contract for {entity}: {reason}")]
    MissingContract {
        /// The entity lacking a contract.
        entity: String,
        /// Reason it is required.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// EncodingError
// ---------------------------------------------------------------------------

/// Errors in SMT / SAT encoding of verification problems.
#[derive(Debug, Error, Clone)]
pub enum EncodingError {
    /// Unsupported constraint type.
    #[error("unsupported constraint: {0}")]
    UnsupportedConstraint(String),

    /// Variable name collision.
    #[error("variable name collision: '{name}' already defined")]
    VariableCollision {
        /// The colliding variable name.
        name: String,
    },

    /// Encoding exceeds solver capacity.
    #[error("encoding too large: {num_clauses} clauses (limit: {limit})")]
    EncodingTooLarge {
        /// Number of clauses generated.
        num_clauses: usize,
        /// Solver limit.
        limit: usize,
    },

    /// Backend solver returned an unexpected result.
    #[error("solver error: {0}")]
    SolverError(String),

    /// Quantifier elimination failed.
    #[error("quantifier elimination failed for {variable}: {reason}")]
    QuantifierEliminationFailed {
        /// The variable.
        variable: String,
        /// Reason.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// ConflictError
// ---------------------------------------------------------------------------

/// Errors in drug–drug interaction conflict detection.
#[derive(Debug, Error, Clone)]
pub enum ConflictError {
    /// Unknown drug in interaction database.
    #[error("unknown drug in interaction lookup: {0}")]
    UnknownDrug(String),

    /// Interaction database is unavailable.
    #[error("interaction database unavailable: {0}")]
    DatabaseUnavailable(String),

    /// Conflict detection internal failure.
    #[error("conflict detection failure: {0}")]
    DetectionFailure(String),

    /// Severity classification error.
    #[error("cannot classify severity for interaction {interaction_id}: {reason}")]
    SeverityClassificationError {
        /// Interaction identifier.
        interaction_id: String,
        /// Reason for failure.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// RecommendationError
// ---------------------------------------------------------------------------

/// Errors in generating clinical recommendations.
#[derive(Debug, Error, Clone)]
pub enum RecommendationError {
    /// No viable alternative found.
    #[error("no alternative found for {drug}: {reason}")]
    NoAlternative {
        /// The drug needing replacement.
        drug: String,
        /// Reason no alternative was found.
        reason: String,
    },

    /// Recommendation conflicts with existing therapy.
    #[error("recommendation conflicts with existing therapy: {detail}")]
    ConflictsWithExisting {
        /// Detail of the conflict.
        detail: String,
    },

    /// Insufficient data to generate recommendation.
    #[error("insufficient data for recommendation: {0}")]
    InsufficientData(String),
}

// ---------------------------------------------------------------------------
// ClinicalError
// ---------------------------------------------------------------------------

/// Errors in clinical data processing.
#[derive(Debug, Error, Clone)]
pub enum ClinicalError {
    /// Missing required patient data.
    #[error("missing patient data: {field}")]
    MissingPatientData {
        /// The missing field.
        field: String,
    },

    /// Lab value out of physiological range.
    #[error("lab value '{name}' = {value} is out of physiological range")]
    LabValueOutOfRange {
        /// Lab value name.
        name: String,
        /// The value.
        value: f64,
    },

    /// Invalid ICD-10 code.
    #[error("invalid ICD-10 code: {0}")]
    InvalidIcd10Code(String),

    /// Inconsistent clinical state (e.g., contradictory conditions).
    #[error("inconsistent clinical state: {0}")]
    InconsistentState(String),

    /// Unknown clinical condition.
    #[error("unknown clinical condition: {0}")]
    UnknownCondition(String),
}

// ---------------------------------------------------------------------------
// ConfigError
// ---------------------------------------------------------------------------

/// Errors in configuration loading and validation.
#[derive(Debug, Error, Clone)]
pub enum ConfigError {
    /// Missing required configuration key.
    #[error("missing configuration key: {0}")]
    MissingKey(String),

    /// Invalid configuration value.
    #[error("invalid value for '{key}': {reason}")]
    InvalidValue {
        /// Configuration key.
        key: String,
        /// Why it is invalid.
        reason: String,
    },

    /// Configuration file not found.
    #[error("configuration file not found: {0}")]
    FileNotFound(String),

    /// Configuration file parse failure.
    #[error("configuration file parse error: {0}")]
    ParseFailure(String),

    /// Conflicting configuration options.
    #[error("conflicting options: '{a}' and '{b}' cannot both be set")]
    ConflictingOptions {
        /// First option.
        a: String,
        /// Second option.
        b: String,
    },
}

// ---------------------------------------------------------------------------
// Conversions from standard library errors
// ---------------------------------------------------------------------------

impl From<std::io::Error> for GuardPharmaError {
    fn from(e: std::io::Error) -> Self {
        GuardPharmaError::Io {
            message: e.to_string(),
        }
    }
}

impl From<serde_json::Error> for GuardPharmaError {
    fn from(e: serde_json::Error) -> Self {
        GuardPharmaError::Serialization {
            message: e.to_string(),
        }
    }
}

impl From<String> for GuardPharmaError {
    fn from(s: String) -> Self {
        GuardPharmaError::Internal {
            message: s,
            context: None,
        }
    }
}

impl From<&str> for GuardPharmaError {
    fn from(s: &str) -> Self {
        GuardPharmaError::Internal {
            message: s.to_string(),
            context: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

impl GuardPharmaError {
    /// Create an internal error with context.
    pub fn internal(message: impl Into<String>, context: impl Into<String>) -> Self {
        GuardPharmaError::Internal {
            message: message.into(),
            context: Some(context.into()),
        }
    }

    /// Create a timeout error.
    pub fn timeout(operation: impl Into<String>, duration_secs: f64) -> Self {
        GuardPharmaError::Timeout {
            operation: operation.into(),
            duration_secs,
        }
    }

    /// Returns `true` if this error represents a timeout.
    pub fn is_timeout(&self) -> bool {
        matches!(self, GuardPharmaError::Timeout { .. })
    }

    /// Returns `true` if this error represents a verification failure
    /// (as opposed to an infrastructure / data error).
    pub fn is_verification_error(&self) -> bool {
        matches!(self, GuardPharmaError::Verification(_))
    }

    /// Returns `true` if this is a contract-related error.
    pub fn is_contract_error(&self) -> bool {
        matches!(self, GuardPharmaError::Contract(_))
    }

    /// Returns `true` for errors that may be transient and worth retrying.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            GuardPharmaError::Timeout { .. } | GuardPharmaError::Io { .. }
        )
    }
}

impl ParseError {
    /// Create a generic parse error.
    pub fn generic(msg: impl Into<String>) -> Self {
        ParseError::Generic(msg.into())
    }

    /// Create a missing-field error.
    pub fn missing_field(field: impl Into<String>, context: impl Into<String>) -> Self {
        ParseError::MissingField {
            field: field.into(),
            context: context.into(),
        }
    }
}

impl VerificationError {
    /// Create a malformed-property error.
    pub fn malformed_property(msg: impl Into<String>) -> Self {
        VerificationError::MalformedProperty(msg.into())
    }
}

// thiserror already provides a Display impl via #[error(...)] attributes.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_display() {
        let e = ParseError::InvalidDrugId("XYZ-999".into());
        assert!(e.to_string().contains("XYZ-999"));
    }

    #[test]
    fn test_parse_error_missing_field() {
        let e = ParseError::missing_field("dose", "warfarin schedule");
        let s = e.to_string();
        assert!(s.contains("dose"));
        assert!(s.contains("warfarin schedule"));
    }

    #[test]
    fn test_pk_model_error_display() {
        let e = PkModelError::MissingParameters {
            drug_id: "warfarin".into(),
        };
        assert!(e.to_string().contains("warfarin"));
    }

    #[test]
    fn test_verification_error_convergence() {
        let e = VerificationError::ConvergenceFailed { iterations: 100 };
        assert!(e.to_string().contains("100"));
    }

    #[test]
    fn test_contract_error_incompatible() {
        let e = ContractError::Incompatible {
            a: "C1".into(),
            b: "C2".into(),
            reason: "overlap".into(),
        };
        let s = e.to_string();
        assert!(s.contains("C1"));
        assert!(s.contains("C2"));
    }

    #[test]
    fn test_encoding_error_display() {
        let e = EncodingError::UnsupportedConstraint("nonlinear".into());
        assert!(e.to_string().contains("nonlinear"));
    }

    #[test]
    fn test_conflict_error_display() {
        let e = ConflictError::UnknownDrug("mystery_drug".into());
        assert!(e.to_string().contains("mystery_drug"));
    }

    #[test]
    fn test_recommendation_error_display() {
        let e = RecommendationError::NoAlternative {
            drug: "warfarin".into(),
            reason: "all alternatives also interact".into(),
        };
        assert!(e.to_string().contains("warfarin"));
    }

    #[test]
    fn test_clinical_error_display() {
        let e = ClinicalError::MissingPatientData {
            field: "eGFR".into(),
        };
        assert!(e.to_string().contains("eGFR"));
    }

    #[test]
    fn test_config_error_display() {
        let e = ConfigError::MissingKey("timeout_secs".into());
        assert!(e.to_string().contains("timeout_secs"));
    }

    #[test]
    fn test_guardpharma_error_from_parse() {
        let pe = ParseError::Generic("bad input".into());
        let ge: GuardPharmaError = pe.into();
        assert!(matches!(ge, GuardPharmaError::Parse(_)));
        assert!(ge.to_string().contains("bad input"));
    }

    #[test]
    fn test_guardpharma_error_from_string() {
        let ge: GuardPharmaError = "something went wrong".into();
        assert!(matches!(ge, GuardPharmaError::Internal { .. }));
    }

    #[test]
    fn test_guardpharma_error_internal_with_context() {
        let ge = GuardPharmaError::internal("oops", "pk_model::solve");
        match &ge {
            GuardPharmaError::Internal { context, .. } => {
                assert_eq!(context.as_deref(), Some("pk_model::solve"));
            }
            _ => panic!("expected Internal variant"),
        }
    }

    #[test]
    fn test_is_timeout() {
        let ge = GuardPharmaError::timeout("model checking", 30.0);
        assert!(ge.is_timeout());
        assert!(!ge.is_verification_error());
        assert!(ge.is_retryable());
    }

    #[test]
    fn test_is_verification_error() {
        let ve = VerificationError::MalformedProperty("bad".into());
        let ge: GuardPharmaError = ve.into();
        assert!(ge.is_verification_error());
        assert!(!ge.is_timeout());
    }

    #[test]
    fn test_is_contract_error() {
        let ce = ContractError::MissingContract {
            entity: "CYP3A4".into(),
            reason: "no contract provided".into(),
        };
        let ge: GuardPharmaError = ce.into();
        assert!(ge.is_contract_error());
    }

    #[test]
    fn test_config_conflicting_options() {
        let ce = ConfigError::ConflictingOptions {
            a: "use_contracts".into(),
            b: "monolithic_mode".into(),
        };
        let s = ce.to_string();
        assert!(s.contains("use_contracts"));
        assert!(s.contains("monolithic_mode"));
    }
}

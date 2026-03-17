//! Core types for the Penumbra floating-point diagnosis and repair engine.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Unique identifier for traced operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpId(pub Uuid);

impl OpId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for OpId {
    fn default() -> Self {
        Self::new()
    }
}

/// Floating-point precision level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precision {
    Half,
    Single,
    Double,
    Quad,
    Extended(u32),
}

/// Rounding mode for floating-point operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoundingMode {
    NearestEven,
    TowardZero,
    TowardPositive,
    TowardNegative,
}

/// Represents a floating-point operation kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FpOperation {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
    Fma,
    Abs,
    Neg,
    Min,
    Max,
    Floor,
    Ceil,
    Round,
    Trunc,
    Exp,
    Log,
    Log2,
    Log10,
    Pow,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Sinh,
    Cosh,
    Tanh,
    Hypot,
    /// A named custom operation.
    Custom(String),
}

impl FpOperation {
    /// Returns the number of input operands expected.
    pub fn arity(&self) -> usize {
        match self {
            FpOperation::Neg | FpOperation::Abs | FpOperation::Sqrt
            | FpOperation::Floor | FpOperation::Ceil | FpOperation::Round
            | FpOperation::Trunc | FpOperation::Exp | FpOperation::Log
            | FpOperation::Log2 | FpOperation::Log10 | FpOperation::Sin
            | FpOperation::Cos | FpOperation::Tan | FpOperation::Asin
            | FpOperation::Acos | FpOperation::Atan | FpOperation::Sinh
            | FpOperation::Cosh | FpOperation::Tanh => 1,
            FpOperation::Add | FpOperation::Sub | FpOperation::Mul
            | FpOperation::Div | FpOperation::Pow | FpOperation::Min
            | FpOperation::Max | FpOperation::Hypot | FpOperation::Atan2 => 2,
            FpOperation::Fma => 3,
            FpOperation::Custom(_) => 0,
        }
    }

    /// Whether the operation is an elementary (hardware-level) operation.
    pub fn is_elementary(&self) -> bool {
        matches!(
            self,
            FpOperation::Add | FpOperation::Sub | FpOperation::Mul
            | FpOperation::Div | FpOperation::Sqrt | FpOperation::Fma
        )
    }
}

impl fmt::Display for FpOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FpOperation::Add => write!(f, "add"),
            FpOperation::Sub => write!(f, "sub"),
            FpOperation::Mul => write!(f, "mul"),
            FpOperation::Div => write!(f, "div"),
            FpOperation::Sqrt => write!(f, "sqrt"),
            FpOperation::Fma => write!(f, "fma"),
            FpOperation::Abs => write!(f, "abs"),
            FpOperation::Neg => write!(f, "neg"),
            FpOperation::Min => write!(f, "min"),
            FpOperation::Max => write!(f, "max"),
            FpOperation::Floor => write!(f, "floor"),
            FpOperation::Ceil => write!(f, "ceil"),
            FpOperation::Round => write!(f, "round"),
            FpOperation::Trunc => write!(f, "trunc"),
            FpOperation::Exp => write!(f, "exp"),
            FpOperation::Log => write!(f, "log"),
            FpOperation::Log2 => write!(f, "log2"),
            FpOperation::Log10 => write!(f, "log10"),
            FpOperation::Pow => write!(f, "pow"),
            FpOperation::Sin => write!(f, "sin"),
            FpOperation::Cos => write!(f, "cos"),
            FpOperation::Tan => write!(f, "tan"),
            FpOperation::Asin => write!(f, "asin"),
            FpOperation::Acos => write!(f, "acos"),
            FpOperation::Atan => write!(f, "atan"),
            FpOperation::Atan2 => write!(f, "atan2"),
            FpOperation::Sinh => write!(f, "sinh"),
            FpOperation::Cosh => write!(f, "cosh"),
            FpOperation::Tanh => write!(f, "tanh"),
            FpOperation::Hypot => write!(f, "hypot"),
            FpOperation::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// Severity of a detected floating-point issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// A span identifying a region in source code.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceSpan {
    pub file: String,
    pub line_start: u32,
    pub line_end: u32,
    pub col_start: u32,
    pub col_end: u32,
}

// ============================================================================
// Float classification
// ============================================================================

/// Classification of IEEE 754 floating-point values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatClass {
    PositiveZero,
    NegativeZero,
    PositiveNormal,
    NegativeNormal,
    PositiveDenormal,
    NegativeDenormal,
    PositiveInfinity,
    NegativeInfinity,
    QuietNaN,
    SignalingNaN,
}

impl fmt::Display for FloatClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatClass::PositiveZero => write!(f, "+0"),
            FloatClass::NegativeZero => write!(f, "-0"),
            FloatClass::PositiveNormal => write!(f, "+normal"),
            FloatClass::NegativeNormal => write!(f, "-normal"),
            FloatClass::PositiveDenormal => write!(f, "+denormal"),
            FloatClass::NegativeDenormal => write!(f, "-denormal"),
            FloatClass::PositiveInfinity => write!(f, "+inf"),
            FloatClass::NegativeInfinity => write!(f, "-inf"),
            FloatClass::QuietNaN => write!(f, "qNaN"),
            FloatClass::SignalingNaN => write!(f, "sNaN"),
        }
    }
}

// ============================================================================
// Float format descriptor
// ============================================================================

/// IEEE 754 floating-point format descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FloatFormat {
    Half,
    Single,
    Double,
    Quad,
    Extended80,
}

impl FloatFormat {
    pub fn total_bits(&self) -> u32 {
        match self {
            FloatFormat::Half => 16,
            FloatFormat::Single => 32,
            FloatFormat::Double => 64,
            FloatFormat::Quad => 128,
            FloatFormat::Extended80 => 80,
        }
    }

    pub fn significand_bits(&self) -> u32 {
        match self {
            FloatFormat::Half => 11,
            FloatFormat::Single => 24,
            FloatFormat::Double => 53,
            FloatFormat::Quad => 113,
            FloatFormat::Extended80 => 64,
        }
    }

    pub fn exponent_bits(&self) -> u32 {
        match self {
            FloatFormat::Half => 5,
            FloatFormat::Single => 8,
            FloatFormat::Double => 11,
            FloatFormat::Quad => 15,
            FloatFormat::Extended80 => 15,
        }
    }

    pub fn exponent_bias(&self) -> i32 {
        (1 << (self.exponent_bits() - 1)) - 1
    }

    pub fn machine_epsilon(&self) -> f64 {
        let p = self.significand_bits() as i32;
        f64::powi(2.0, 1 - p)
    }
}

impl fmt::Display for FloatFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatFormat::Half => write!(f, "binary16"),
            FloatFormat::Single => write!(f, "binary32"),
            FloatFormat::Double => write!(f, "binary64"),
            FloatFormat::Quad => write!(f, "binary128"),
            FloatFormat::Extended80 => write!(f, "extended80"),
        }
    }
}

// ============================================================================
// Diagnosis categories
// ============================================================================

/// Categories of floating-point error patterns that Penumbra can diagnose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiagnosisCategory {
    CatastrophicCancellation,
    BenignCancellation,
    Absorption,
    Smearing,
    AmplifiedRounding,
    Overflow,
    Underflow,
    TrailingDigitLoss,
    AccumulatedRounding,
    DenormalError,
    InvalidOperation,
    DivisionInstability,
    NoError,
}

impl DiagnosisCategory {
    pub fn description(&self) -> &'static str {
        match self {
            DiagnosisCategory::CatastrophicCancellation =>
                "Catastrophic cancellation: subtraction of nearly equal values",
            DiagnosisCategory::BenignCancellation =>
                "Benign cancellation: cancellation that does not significantly amplify errors",
            DiagnosisCategory::Absorption =>
                "Absorption: small operand absorbed into large operand",
            DiagnosisCategory::Smearing =>
                "Smearing: partial loss of significance during summation",
            DiagnosisCategory::AmplifiedRounding =>
                "Amplified rounding: rounding error magnified by ill-conditioning",
            DiagnosisCategory::Overflow =>
                "Overflow: computed value exceeds representable range",
            DiagnosisCategory::Underflow =>
                "Underflow: computed value too small for normal representation",
            DiagnosisCategory::TrailingDigitLoss =>
                "Trailing digit loss: low-order bits lost",
            DiagnosisCategory::AccumulatedRounding =>
                "Accumulated rounding: aggregate errors from many operations",
            DiagnosisCategory::DenormalError =>
                "Denormal error: errors from subnormal arithmetic",
            DiagnosisCategory::InvalidOperation =>
                "Invalid operation: computation producing NaN",
            DiagnosisCategory::DivisionInstability =>
                "Division instability: near-zero divisor amplifying errors",
            DiagnosisCategory::NoError =>
                "No significant floating-point error detected",
        }
    }

    pub fn typical_severity(&self) -> f64 {
        match self {
            DiagnosisCategory::NoError => 0.0,
            DiagnosisCategory::BenignCancellation => 0.1,
            DiagnosisCategory::TrailingDigitLoss => 0.2,
            DiagnosisCategory::DenormalError => 0.3,
            DiagnosisCategory::AccumulatedRounding => 0.4,
            DiagnosisCategory::Smearing => 0.5,
            DiagnosisCategory::Absorption => 0.6,
            DiagnosisCategory::AmplifiedRounding => 0.7,
            DiagnosisCategory::DivisionInstability => 0.8,
            DiagnosisCategory::Underflow => 0.8,
            DiagnosisCategory::Overflow => 0.9,
            DiagnosisCategory::CatastrophicCancellation => 0.95,
            DiagnosisCategory::InvalidOperation => 1.0,
        }
    }
}

impl fmt::Display for DiagnosisCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ============================================================================
// Error type
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum PenumbraError {
    #[error("IEEE 754 error: {0}")]
    Ieee754Error(String),
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Computation overflow")]
    ComputationOverflow,
    #[error("Computation underflow")]
    ComputationUnderflow,
    #[error("Division by zero")]
    DivisionByZero,
}

pub type PenumbraResult<T> = Result<T, PenumbraError>;

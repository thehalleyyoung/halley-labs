//! Mutation operator definitions and mutant tracking for the MutSpec system.
//!
//! Defines the catalogue of mutation operators (AOR, ROR, LCR, UOI, …),
//! mutation sites, mutant descriptors, and kill-tracking information.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::errors::SpanInfo;
use crate::types::QfLiaType;

// ---------------------------------------------------------------------------
// MutationOperator
// ---------------------------------------------------------------------------

/// A mutation operator that can be applied to a program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MutationOperator {
    /// Arithmetic Operator Replacement: replace +, -, *, /, % with each other.
    Aor,
    /// Relational Operator Replacement: replace <, <=, ==, !=, >=, > with each other.
    Ror,
    /// Logical Connector Replacement: replace &&, || with each other or with operands.
    Lcr,
    /// Unary Operator Insertion: insert/remove negation on numeric or boolean exprs.
    Uoi,
    /// Absolute Value Insertion: wrap numeric expression in |x| or -|x|.
    Abs,
    /// Conditional Operator Replacement: replace ternary condition with true/false.
    Cor,
    /// Statement Deletion: remove a statement from the program.
    Sdl,
    /// Return Value Replacement: replace return expression with default/boundary values.
    Rvr,
    /// Constant Replacement: replace constants (0, 1, -1, boundary values).
    Crc,
    /// Array Index Replacement: modify array indices by +/-1.
    Air,
    /// Operand Swap: swap LHS and RHS of binary operators.
    Osw,
    /// Branch Condition Negation: negate an if-condition.
    Bcn,
}

impl MutationOperator {
    /// Short mnemonic.
    pub fn mnemonic(&self) -> &'static str {
        match self {
            MutationOperator::Aor => "AOR",
            MutationOperator::Ror => "ROR",
            MutationOperator::Lcr => "LCR",
            MutationOperator::Uoi => "UOI",
            MutationOperator::Abs => "ABS",
            MutationOperator::Cor => "COR",
            MutationOperator::Sdl => "SDL",
            MutationOperator::Rvr => "RVR",
            MutationOperator::Crc => "CRC",
            MutationOperator::Air => "AIR",
            MutationOperator::Osw => "OSW",
            MutationOperator::Bcn => "BCN",
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            MutationOperator::Aor => "Arithmetic Operator Replacement",
            MutationOperator::Ror => "Relational Operator Replacement",
            MutationOperator::Lcr => "Logical Connector Replacement",
            MutationOperator::Uoi => "Unary Operator Insertion",
            MutationOperator::Abs => "Absolute Value Insertion",
            MutationOperator::Cor => "Conditional Operator Replacement",
            MutationOperator::Sdl => "Statement Deletion",
            MutationOperator::Rvr => "Return Value Replacement",
            MutationOperator::Crc => "Constant Replacement",
            MutationOperator::Air => "Array Index Replacement",
            MutationOperator::Osw => "Operand Swap",
            MutationOperator::Bcn => "Branch Condition Negation",
        }
    }

    /// One-line description.
    pub fn description(&self) -> &'static str {
        match self {
            MutationOperator::Aor => "Replace arithmetic operators (+, -, *, /, %) with each other",
            MutationOperator::Ror => {
                "Replace relational operators (<, <=, ==, !=, >=, >) with each other"
            }
            MutationOperator::Lcr => {
                "Replace logical connectors (&&, ||) with each other or operands"
            }
            MutationOperator::Uoi => "Insert or remove unary negation on expressions",
            MutationOperator::Abs => {
                "Wrap numeric expression in absolute value or negated absolute value"
            }
            MutationOperator::Cor => {
                "Replace ternary/conditional operator with true or false branch"
            }
            MutationOperator::Sdl => "Delete individual statements from the program",
            MutationOperator::Rvr => "Replace return value with default or boundary values",
            MutationOperator::Crc => "Replace constants with related values (0, 1, -1, boundaries)",
            MutationOperator::Air => "Modify array access indices by +1 or -1",
            MutationOperator::Osw => "Swap operands of binary operators",
            MutationOperator::Bcn => "Negate the condition of an if statement",
        }
    }

    /// Which expression/statement types this operator applies to.
    pub fn applicable_contexts(&self) -> &'static [&'static str] {
        match self {
            MutationOperator::Aor => &["BinaryArith"],
            MutationOperator::Ror => &["Relational"],
            MutationOperator::Lcr => &["LogicalAnd", "LogicalOr"],
            MutationOperator::Uoi => &["UnaryArith", "LogicalNot", "Var"],
            MutationOperator::Abs => &["BinaryArith", "Var", "IntLiteral"],
            MutationOperator::Cor => &["Conditional"],
            MutationOperator::Sdl => &["Assign", "Assert", "VarDecl"],
            MutationOperator::Rvr => &["Return"],
            MutationOperator::Crc => &["IntLiteral", "BoolLiteral"],
            MutationOperator::Air => &["ArrayAccess"],
            MutationOperator::Osw => &["BinaryArith", "Relational"],
            MutationOperator::Bcn => &["IfElse"],
        }
    }

    /// The "standard" operators used in most mutation testing studies.
    pub fn standard_set() -> Vec<MutationOperator> {
        vec![
            MutationOperator::Aor,
            MutationOperator::Ror,
            MutationOperator::Lcr,
            MutationOperator::Uoi,
            MutationOperator::Sdl,
        ]
    }

    /// All available operators.
    pub fn all() -> Vec<MutationOperator> {
        vec![
            MutationOperator::Aor,
            MutationOperator::Ror,
            MutationOperator::Lcr,
            MutationOperator::Uoi,
            MutationOperator::Abs,
            MutationOperator::Cor,
            MutationOperator::Sdl,
            MutationOperator::Rvr,
            MutationOperator::Crc,
            MutationOperator::Air,
            MutationOperator::Osw,
            MutationOperator::Bcn,
        ]
    }

    /// Parse from mnemonic string (case-insensitive).
    pub fn from_mnemonic(s: &str) -> Option<MutationOperator> {
        match s.to_uppercase().as_str() {
            "AOR" => Some(MutationOperator::Aor),
            "ROR" => Some(MutationOperator::Ror),
            "LCR" => Some(MutationOperator::Lcr),
            "UOI" => Some(MutationOperator::Uoi),
            "ABS" => Some(MutationOperator::Abs),
            "COR" => Some(MutationOperator::Cor),
            "SDL" => Some(MutationOperator::Sdl),
            "RVR" => Some(MutationOperator::Rvr),
            "CRC" => Some(MutationOperator::Crc),
            "AIR" => Some(MutationOperator::Air),
            "OSW" => Some(MutationOperator::Osw),
            "BCN" => Some(MutationOperator::Bcn),
            _ => None,
        }
    }

    /// Returns true if this operator applies to expressions (vs statements).
    pub fn is_expression_level(&self) -> bool {
        !matches!(
            self,
            MutationOperator::Sdl | MutationOperator::Rvr | MutationOperator::Bcn
        )
    }

    /// Returns true if this is a "sufficient" operator for mutation testing according to
    /// common selective-mutation studies (AOR, ROR, LCR, UOI, ABS).
    pub fn is_sufficient(&self) -> bool {
        matches!(
            self,
            MutationOperator::Aor
                | MutationOperator::Ror
                | MutationOperator::Lcr
                | MutationOperator::Uoi
                | MutationOperator::Abs
        )
    }

    /// Returns the applicable QfLia types for the operands this operator touches.
    pub fn applicable_types(&self) -> Vec<QfLiaType> {
        match self {
            MutationOperator::Aor | MutationOperator::Abs | MutationOperator::Air => {
                vec![QfLiaType::Int, QfLiaType::Long]
            }
            MutationOperator::Ror => vec![QfLiaType::Int, QfLiaType::Long, QfLiaType::Boolean],
            MutationOperator::Lcr | MutationOperator::Uoi | MutationOperator::Bcn => {
                vec![QfLiaType::Boolean]
            }
            _ => QfLiaType::all_value_types().to_vec(),
        }
    }
}

impl fmt::Display for MutationOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mnemonic())
    }
}

// ---------------------------------------------------------------------------
// MutantId
// ---------------------------------------------------------------------------

/// Unique identifier for a mutant, wrapping a UUID.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MutantId(pub Uuid);

impl MutantId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Short 8-character hex prefix for display.
    pub fn short(&self) -> String {
        self.0.to_string()[..8].to_string()
    }
}

impl Default for MutantId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MutantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// MutationSite
// ---------------------------------------------------------------------------

/// A specific location in the program where a mutation can be applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationSite {
    pub location: SpanInfo,
    pub operator: MutationOperator,
    pub original: String,
    pub replacement: String,
    pub function_name: Option<String>,
    pub expression_type: Option<QfLiaType>,
}

impl MutationSite {
    pub fn new(
        location: SpanInfo,
        operator: MutationOperator,
        original: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        Self {
            location,
            operator,
            original: original.into(),
            replacement: replacement.into(),
            function_name: None,
            expression_type: None,
        }
    }

    pub fn with_function(mut self, name: impl Into<String>) -> Self {
        self.function_name = Some(name.into());
        self
    }

    pub fn with_type(mut self, ty: QfLiaType) -> Self {
        self.expression_type = Some(ty);
        self
    }

    /// Short summary for display.
    pub fn summary(&self) -> String {
        format!(
            "{}: `{}` -> `{}`",
            self.operator, self.original, self.replacement
        )
    }
}

impl fmt::Display for MutationSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{} at {}] `{}` -> `{}`",
            self.operator, self.location, self.original, self.replacement
        )?;
        if let Some(ref func) = self.function_name {
            write!(f, " in {func}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MutantStatus / KillInfo
// ---------------------------------------------------------------------------

/// Status of a mutant after analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MutantStatus {
    /// Still alive: no distinguishing specification found.
    Alive,
    /// Killed by a distinguishing specification.
    Killed,
    /// Proven semantically equivalent to the original.
    Equivalent,
    /// Analysis timed out.
    Timeout,
    /// An error occurred during analysis.
    Error(String),
}

impl MutantStatus {
    pub fn is_alive(&self) -> bool {
        matches!(self, MutantStatus::Alive)
    }

    pub fn is_killed(&self) -> bool {
        matches!(self, MutantStatus::Killed)
    }

    pub fn is_equivalent(&self) -> bool {
        matches!(self, MutantStatus::Equivalent)
    }

    pub fn is_terminal(&self) -> bool {
        !matches!(self, MutantStatus::Alive)
    }

    pub fn is_error(&self) -> bool {
        matches!(self, MutantStatus::Error(_))
    }
}

impl fmt::Display for MutantStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MutantStatus::Alive => write!(f, "ALIVE"),
            MutantStatus::Killed => write!(f, "KILLED"),
            MutantStatus::Equivalent => write!(f, "EQUIVALENT"),
            MutantStatus::Timeout => write!(f, "TIMEOUT"),
            MutantStatus::Error(msg) => write!(f, "ERROR({msg})"),
        }
    }
}

/// Information about how a mutant was killed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KillInfo {
    pub test_name: String,
    pub execution_time_ms: f64,
    pub distinguishing_input: Option<String>,
    pub contract_clause: Option<String>,
}

impl KillInfo {
    pub fn new(test_name: impl Into<String>, execution_time_ms: f64) -> Self {
        Self {
            test_name: test_name.into(),
            execution_time_ms,
            distinguishing_input: None,
            contract_clause: None,
        }
    }

    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.distinguishing_input = Some(input.into());
        self
    }

    pub fn with_clause(mut self, clause: impl Into<String>) -> Self {
        self.contract_clause = Some(clause.into());
        self
    }
}

impl fmt::Display for KillInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "killed by `{}` in {:.1}ms",
            self.test_name, self.execution_time_ms
        )?;
        if let Some(ref input) = self.distinguishing_input {
            write!(f, " (input: {input})")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MutantDescriptor
// ---------------------------------------------------------------------------

/// Complete description of a single mutant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MutantDescriptor {
    pub id: MutantId,
    pub operator: MutationOperator,
    pub site: MutationSite,
    pub status: MutantStatus,
    pub kill_info: Option<KillInfo>,
    pub equivalent_reason: Option<String>,
}

impl MutantDescriptor {
    pub fn new(operator: MutationOperator, site: MutationSite) -> Self {
        Self {
            id: MutantId::new(),
            operator,
            site,
            status: MutantStatus::Alive,
            kill_info: None,
            equivalent_reason: None,
        }
    }

    pub fn with_id(mut self, id: MutantId) -> Self {
        self.id = id;
        self
    }

    pub fn mark_killed(&mut self, kill_info: KillInfo) {
        self.status = MutantStatus::Killed;
        self.kill_info = Some(kill_info);
    }

    pub fn mark_equivalent(&mut self, reason: impl Into<String>) {
        self.status = MutantStatus::Equivalent;
        self.equivalent_reason = Some(reason.into());
    }

    pub fn mark_timeout(&mut self) {
        self.status = MutantStatus::Timeout;
    }

    pub fn mark_error(&mut self, msg: impl Into<String>) {
        self.status = MutantStatus::Error(msg.into());
    }

    pub fn is_alive(&self) -> bool {
        self.status.is_alive()
    }

    pub fn is_killed(&self) -> bool {
        self.status.is_killed()
    }

    /// Short summary line.
    pub fn summary(&self) -> String {
        format!(
            "[{}] {} {} — {}",
            self.id.short(),
            self.operator,
            self.site.summary(),
            self.status
        )
    }
}

impl fmt::Display for MutantDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Mutant {}", self.id)?;
        writeln!(
            f,
            "  Operator: {} ({})",
            self.operator,
            self.operator.name()
        )?;
        writeln!(f, "  Site:     {}", self.site)?;
        writeln!(f, "  Status:   {}", self.status)?;
        if let Some(ref ki) = self.kill_info {
            writeln!(f, "  Kill:     {ki}")?;
        }
        if let Some(ref reason) = self.equivalent_reason {
            writeln!(f, "  Equiv:    {reason}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> SpanInfo {
        SpanInfo::unknown()
    }

    #[test]
    fn test_operator_mnemonic() {
        assert_eq!(MutationOperator::Aor.mnemonic(), "AOR");
        assert_eq!(MutationOperator::Ror.mnemonic(), "ROR");
        assert_eq!(MutationOperator::Bcn.mnemonic(), "BCN");
    }

    #[test]
    fn test_operator_from_mnemonic() {
        assert_eq!(
            MutationOperator::from_mnemonic("AOR"),
            Some(MutationOperator::Aor)
        );
        assert_eq!(
            MutationOperator::from_mnemonic("ror"),
            Some(MutationOperator::Ror)
        );
        assert_eq!(MutationOperator::from_mnemonic("XYZ"), None);
    }

    #[test]
    fn test_operator_display() {
        assert_eq!(MutationOperator::Lcr.to_string(), "LCR");
    }

    #[test]
    fn test_operator_standard_set() {
        let std = MutationOperator::standard_set();
        assert!(std.contains(&MutationOperator::Aor));
        assert!(std.contains(&MutationOperator::Ror));
        assert!(!std.contains(&MutationOperator::Cor));
    }

    #[test]
    fn test_operator_all() {
        let all = MutationOperator::all();
        assert_eq!(all.len(), 12);
    }

    #[test]
    fn test_operator_is_expression_level() {
        assert!(MutationOperator::Aor.is_expression_level());
        assert!(!MutationOperator::Sdl.is_expression_level());
        assert!(!MutationOperator::Bcn.is_expression_level());
    }

    #[test]
    fn test_operator_is_sufficient() {
        assert!(MutationOperator::Aor.is_sufficient());
        assert!(!MutationOperator::Sdl.is_sufficient());
    }

    #[test]
    fn test_operator_applicable_types() {
        let types = MutationOperator::Aor.applicable_types();
        assert!(types.contains(&QfLiaType::Int));
        assert!(!types.contains(&QfLiaType::Boolean));
    }

    #[test]
    fn test_operator_applicable_contexts() {
        let ctxs = MutationOperator::Aor.applicable_contexts();
        assert!(ctxs.contains(&"BinaryArith"));
    }

    #[test]
    fn test_operator_name_description() {
        let op = MutationOperator::Ror;
        assert!(!op.name().is_empty());
        assert!(!op.description().is_empty());
    }

    #[test]
    fn test_mutant_id() {
        let id = MutantId::new();
        assert_eq!(id.short().len(), 8);
        let s = id.to_string();
        assert!(s.len() > 8);
    }

    #[test]
    fn test_mutant_id_default() {
        let id1 = MutantId::default();
        let id2 = MutantId::default();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_mutation_site() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Aor, "a + b", "a - b")
            .with_function("foo")
            .with_type(QfLiaType::Int);
        assert_eq!(site.function_name.as_deref(), Some("foo"));
        assert_eq!(site.expression_type, Some(QfLiaType::Int));
        let summary = site.summary();
        assert!(summary.contains("AOR"));
        assert!(summary.contains("a + b"));
    }

    #[test]
    fn test_mutation_site_display() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Ror, "a < b", "a <= b")
            .with_function("bar");
        let s = site.to_string();
        assert!(s.contains("ROR"));
        assert!(s.contains("bar"));
    }

    #[test]
    fn test_mutant_status() {
        assert!(MutantStatus::Alive.is_alive());
        assert!(MutantStatus::Killed.is_killed());
        assert!(MutantStatus::Equivalent.is_equivalent());
        assert!(MutantStatus::Error("x".into()).is_error());
        assert!(!MutantStatus::Alive.is_terminal());
        assert!(MutantStatus::Killed.is_terminal());
    }

    #[test]
    fn test_mutant_status_display() {
        assert_eq!(MutantStatus::Alive.to_string(), "ALIVE");
        assert_eq!(MutantStatus::Killed.to_string(), "KILLED");
        assert_eq!(MutantStatus::Equivalent.to_string(), "EQUIVALENT");
        assert_eq!(MutantStatus::Timeout.to_string(), "TIMEOUT");
        assert!(MutantStatus::Error("fail".into())
            .to_string()
            .contains("fail"));
    }

    #[test]
    fn test_kill_info() {
        let ki = KillInfo::new("test_add", 12.5)
            .with_input("a=1, b=2")
            .with_clause("ensures result > 0");
        let s = ki.to_string();
        assert!(s.contains("test_add"));
        assert!(s.contains("12.5"));
        assert!(s.contains("a=1"));
    }

    #[test]
    fn test_mutant_descriptor() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Aor, "+", "-");
        let mut desc = MutantDescriptor::new(MutationOperator::Aor, site);
        assert!(desc.is_alive());
        assert!(!desc.is_killed());

        desc.mark_killed(KillInfo::new("test1", 5.0));
        assert!(desc.is_killed());
        assert!(desc.kill_info.is_some());
    }

    #[test]
    fn test_mutant_descriptor_equivalent() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Aor, "+", "-");
        let mut desc = MutantDescriptor::new(MutationOperator::Aor, site);
        desc.mark_equivalent("algebraic identity");
        assert!(desc.status.is_equivalent());
        assert_eq!(
            desc.equivalent_reason.as_deref(),
            Some("algebraic identity")
        );
    }

    #[test]
    fn test_mutant_descriptor_timeout() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Ror, "<", "<=");
        let mut desc = MutantDescriptor::new(MutationOperator::Ror, site);
        desc.mark_timeout();
        assert_eq!(desc.status, MutantStatus::Timeout);
    }

    #[test]
    fn test_mutant_descriptor_error() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Sdl, "stmt", "");
        let mut desc = MutantDescriptor::new(MutationOperator::Sdl, site);
        desc.mark_error("internal failure");
        assert!(desc.status.is_error());
    }

    #[test]
    fn test_mutant_descriptor_summary() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Aor, "+", "*");
        let desc = MutantDescriptor::new(MutationOperator::Aor, site);
        let s = desc.summary();
        assert!(s.contains("AOR"));
        assert!(s.contains("ALIVE"));
    }

    #[test]
    fn test_mutant_descriptor_display() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Ror, ">", ">=");
        let desc = MutantDescriptor::new(MutationOperator::Ror, site);
        let s = desc.to_string();
        assert!(s.contains("Mutant"));
        assert!(s.contains("ROR"));
    }

    #[test]
    fn test_mutant_descriptor_serialization() {
        let site = MutationSite::new(dummy_span(), MutationOperator::Aor, "+", "-");
        let desc = MutantDescriptor::new(MutationOperator::Aor, site);
        let json = serde_json::to_string(&desc).unwrap();
        let desc2: MutantDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(desc.operator, desc2.operator);
        assert_eq!(desc.status, desc2.status);
    }
}

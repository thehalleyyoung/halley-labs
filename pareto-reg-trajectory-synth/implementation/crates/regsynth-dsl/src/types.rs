use regsynth_types::Jurisdiction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Types in the regulatory DSL.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DslType {
    /// Obligation type indexed by jurisdiction: OBL[j], PERM[j], PROH[j]
    ObligationType {
        kind: ObligationTypeKind,
        jurisdiction: Option<String>,
    },
    /// Temporal interval type
    Temporal,
    /// Boolean type
    Bool,
    /// Integer type
    Int,
    /// Float type
    Float,
    /// String type
    Str,
    /// Cost type
    Cost,
    /// Strategy type
    Strategy,
    /// Set of elements of type T
    Set(Box<DslType>),
    /// Function type: (params) -> return
    Function {
        params: Vec<DslType>,
        ret: Box<DslType>,
    },
    /// Risk level type
    RiskLevel,
    /// Formalizability grade type
    FormalizabilityGrade,
    /// Domain type
    Domain,
    /// Article reference type
    ArticleRef,
    /// Unit type (used for declarations that don't produce values)
    Unit,
    /// Type variable for inference
    TypeVar(u32),
    /// Error type (used for error recovery)
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObligationTypeKind {
    Obligation,
    Permission,
    Prohibition,
    /// Any obligation kind (for polymorphic composition)
    Any,
}

impl fmt::Display for DslType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObligationType { kind, jurisdiction } => {
                let k = match kind {
                    ObligationTypeKind::Obligation => "OBL",
                    ObligationTypeKind::Permission => "PERM",
                    ObligationTypeKind::Prohibition => "PROH",
                    ObligationTypeKind::Any => "OBL_ANY",
                };
                if let Some(j) = jurisdiction {
                    write!(f, "{}[{}]", k, j)
                } else {
                    write!(f, "{}", k)
                }
            }
            Self::Temporal => write!(f, "Temporal"),
            Self::Bool => write!(f, "Bool"),
            Self::Int => write!(f, "Int"),
            Self::Float => write!(f, "Float"),
            Self::Str => write!(f, "String"),
            Self::Cost => write!(f, "Cost"),
            Self::Strategy => write!(f, "Strategy"),
            Self::Set(inner) => write!(f, "Set<{}>", inner),
            Self::Function { params, ret } => {
                write!(f, "(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Self::RiskLevel => write!(f, "RiskLevel"),
            Self::FormalizabilityGrade => write!(f, "FormalizabilityGrade"),
            Self::Domain => write!(f, "Domain"),
            Self::ArticleRef => write!(f, "ArticleRef"),
            Self::Unit => write!(f, "Unit"),
            Self::TypeVar(id) => write!(f, "?T{}", id),
            Self::Error => write!(f, "Error"),
        }
    }
}

impl DslType {
    /// Check if this type is numeric (Int or Float).
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Int | Self::Float)
    }

    /// Check if this type is an obligation type.
    pub fn is_obligation(&self) -> bool {
        matches!(self, Self::ObligationType { .. })
    }

    /// Get the result type of a binary arithmetic operation.
    pub fn arithmetic_result(&self, other: &DslType) -> Option<DslType> {
        match (self, other) {
            (Self::Int, Self::Int) => Some(Self::Int),
            (Self::Float, Self::Float) | (Self::Int, Self::Float) | (Self::Float, Self::Int) => {
                Some(Self::Float)
            }
            _ => None,
        }
    }

    /// Check if this type is a comparison operand type.
    pub fn is_comparable(&self) -> bool {
        matches!(
            self,
            Self::Int | Self::Float | Self::Str | Self::Bool | Self::RiskLevel | Self::FormalizabilityGrade
        )
    }
}

/// Type environment for scope-based type checking.
#[derive(Debug, Clone)]
pub struct TypeEnv {
    scopes: Vec<HashMap<String, DslType>>,
    jurisdictions: HashMap<String, JurisdictionInfo>,
    next_type_var: u32,
}

#[derive(Debug, Clone)]
pub struct JurisdictionInfo {
    pub name: String,
    pub parent: Option<String>,
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            jurisdictions: HashMap::new(),
            next_type_var: 0,
        }
    }

    /// Push a new scope.
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the current scope.
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Bind a name to a type in the current scope.
    pub fn bind(&mut self, name: String, ty: DslType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    /// Look up a name, searching from innermost to outermost scope.
    pub fn lookup(&self, name: &str) -> Option<&DslType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// Check if a name is bound in the current (innermost) scope.
    pub fn is_bound_in_current_scope(&self, name: &str) -> bool {
        self.scopes
            .last()
            .map(|s| s.contains_key(name))
            .unwrap_or(false)
    }

    /// Register a jurisdiction.
    pub fn register_jurisdiction(&mut self, name: String, parent: Option<String>) {
        self.jurisdictions.insert(
            name.clone(),
            JurisdictionInfo {
                name,
                parent,
            },
        );
    }

    /// Check if a jurisdiction is registered.
    pub fn has_jurisdiction(&self, name: &str) -> bool {
        self.jurisdictions.contains_key(name)
    }

    /// Get jurisdiction info.
    pub fn get_jurisdiction(&self, name: &str) -> Option<&JurisdictionInfo> {
        self.jurisdictions.get(name)
    }

    /// Generate a fresh type variable.
    pub fn fresh_type_var(&mut self) -> DslType {
        let id = self.next_type_var;
        self.next_type_var += 1;
        DslType::TypeVar(id)
    }

    /// Check subtyping: is `sub` a subtype of `sup`?
    /// The key rule is jurisdiction lattice subtyping: OBL[EU-DE] <: OBL[EU]
    pub fn is_subtype(&self, sub: &DslType, sup: &DslType) -> bool {
        if sub == sup {
            return true;
        }
        // Error type is a subtype of everything (for error recovery)
        if matches!(sub, DslType::Error) || matches!(sup, DslType::Error) {
            return true;
        }
        // Int is a subtype of Float
        if matches!(sub, DslType::Int) && matches!(sup, DslType::Float) {
            return true;
        }
        // Obligation subtyping via jurisdiction lattice
        if let (
            DslType::ObligationType {
                kind: k1,
                jurisdiction: Some(j1),
            },
            DslType::ObligationType {
                kind: k2,
                jurisdiction: Some(j2),
            },
        ) = (sub, sup)
        {
            if k1 == k2 || matches!(k2, ObligationTypeKind::Any) {
                return self.is_jurisdiction_subtype(j1, j2);
            }
        }
        // OBL <: OBL_ANY
        if let (
            DslType::ObligationType { jurisdiction: j1, .. },
            DslType::ObligationType {
                kind: ObligationTypeKind::Any,
                jurisdiction: j2,
            },
        ) = (sub, sup)
        {
            match (j1, j2) {
                (_, None) => return true,
                (Some(j1), Some(j2)) => return self.is_jurisdiction_subtype(j1, j2),
                _ => {}
            }
        }
        // Set covariance
        if let (DslType::Set(inner_sub), DslType::Set(inner_sup)) = (sub, sup) {
            return self.is_subtype(inner_sub, inner_sup);
        }
        false
    }

    /// Check if jurisdiction j1 is a sub-jurisdiction of j2.
    fn is_jurisdiction_subtype(&self, j1: &str, j2: &str) -> bool {
        if j1 == j2 {
            return true;
        }
        let j1_jur = Jurisdiction::new(j1);
        let j2_jur = Jurisdiction::new(j2);
        j2_jur.is_parent_of(&j1_jur)
    }

    /// Compute the join (least upper bound) of two types.
    pub fn type_join(&self, a: &DslType, b: &DslType) -> Option<DslType> {
        if a == b {
            return Some(a.clone());
        }
        if self.is_subtype(a, b) {
            return Some(b.clone());
        }
        if self.is_subtype(b, a) {
            return Some(a.clone());
        }
        // Numeric promotion
        match (a, b) {
            (DslType::Int, DslType::Float) | (DslType::Float, DslType::Int) => {
                Some(DslType::Float)
            }
            _ => None,
        }
    }
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_display() {
        assert_eq!(DslType::Bool.to_string(), "Bool");
        assert_eq!(DslType::Int.to_string(), "Int");
        assert_eq!(
            DslType::ObligationType {
                kind: ObligationTypeKind::Obligation,
                jurisdiction: Some("EU".into()),
            }
            .to_string(),
            "OBL[EU]"
        );
        assert_eq!(DslType::Set(Box::new(DslType::Int)).to_string(), "Set<Int>");
        assert_eq!(
            DslType::Function {
                params: vec![DslType::Int, DslType::Bool],
                ret: Box::new(DslType::Str),
            }
            .to_string(),
            "(Int, Bool) -> String"
        );
    }

    #[test]
    fn test_type_env_scoping() {
        let mut env = TypeEnv::new();
        env.bind("x".into(), DslType::Int);
        assert_eq!(env.lookup("x"), Some(&DslType::Int));

        env.push_scope();
        env.bind("y".into(), DslType::Bool);
        assert_eq!(env.lookup("x"), Some(&DslType::Int));
        assert_eq!(env.lookup("y"), Some(&DslType::Bool));

        env.pop_scope();
        assert_eq!(env.lookup("x"), Some(&DslType::Int));
        assert_eq!(env.lookup("y"), None);
    }

    #[test]
    fn test_subtyping_same() {
        let env = TypeEnv::new();
        assert!(env.is_subtype(&DslType::Int, &DslType::Int));
        assert!(env.is_subtype(&DslType::Bool, &DslType::Bool));
    }

    #[test]
    fn test_subtyping_int_float() {
        let env = TypeEnv::new();
        assert!(env.is_subtype(&DslType::Int, &DslType::Float));
        assert!(!env.is_subtype(&DslType::Float, &DslType::Int));
    }

    #[test]
    fn test_subtyping_jurisdiction_lattice() {
        let mut env = TypeEnv::new();
        env.register_jurisdiction("EU".into(), None);
        env.register_jurisdiction("EU-DE".into(), Some("EU".into()));

        let obl_eu = DslType::ObligationType {
            kind: ObligationTypeKind::Obligation,
            jurisdiction: Some("EU".into()),
        };
        let obl_eu_de = DslType::ObligationType {
            kind: ObligationTypeKind::Obligation,
            jurisdiction: Some("EU-DE".into()),
        };

        // EU-DE is a child of EU, so OBL[EU-DE] <: OBL[EU]
        assert!(env.is_subtype(&obl_eu_de, &obl_eu));
        assert!(!env.is_subtype(&obl_eu, &obl_eu_de));
    }

    #[test]
    fn test_type_join() {
        let env = TypeEnv::new();
        assert_eq!(env.type_join(&DslType::Int, &DslType::Int), Some(DslType::Int));
        assert_eq!(
            env.type_join(&DslType::Int, &DslType::Float),
            Some(DslType::Float)
        );
        assert_eq!(env.type_join(&DslType::Bool, &DslType::Str), None);
    }

    #[test]
    fn test_arithmetic_result() {
        assert_eq!(DslType::Int.arithmetic_result(&DslType::Int), Some(DslType::Int));
        assert_eq!(
            DslType::Int.arithmetic_result(&DslType::Float),
            Some(DslType::Float)
        );
        assert_eq!(DslType::Bool.arithmetic_result(&DslType::Int), None);
    }

    #[test]
    fn test_is_numeric() {
        assert!(DslType::Int.is_numeric());
        assert!(DslType::Float.is_numeric());
        assert!(!DslType::Bool.is_numeric());
    }

    #[test]
    fn test_fresh_type_var() {
        let mut env = TypeEnv::new();
        let t1 = env.fresh_type_var();
        let t2 = env.fresh_type_var();
        assert_ne!(t1, t2);
        assert!(matches!(t1, DslType::TypeVar(0)));
        assert!(matches!(t2, DslType::TypeVar(1)));
    }
}

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LowerLevelType {
    LP,
    QP,
    Convex,
    Nonconvex,
    MixedInteger,
}

impl fmt::Display for LowerLevelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerLevelType::LP => write!(f, "LP"),
            LowerLevelType::QP => write!(f, "QP"),
            LowerLevelType::Convex => write!(f, "Convex"),
            LowerLevelType::Nonconvex => write!(f, "Nonconvex"),
            LowerLevelType::MixedInteger => write!(f, "MixedInteger"),
        }
    }
}

impl LowerLevelType {
    pub fn is_convex(&self) -> bool {
        matches!(self, Self::LP | Self::QP | Self::Convex)
    }
    pub fn supports_kkt(&self) -> bool {
        matches!(self, Self::LP | Self::QP | Self::Convex)
    }
    pub fn supports_strong_duality(&self) -> bool {
        matches!(self, Self::LP)
    }
    pub fn requires_value_function(&self) -> bool {
        matches!(self, Self::MixedInteger | Self::Nonconvex)
    }
    pub fn difficulty_score(&self) -> u32 {
        match self {
            Self::LP => 1,
            Self::QP => 2,
            Self::Convex => 3,
            Self::Nonconvex => 4,
            Self::MixedInteger => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CouplingType {
    ObjectiveOnly,
    ConstraintOnly,
    Both,
    None,
}

impl fmt::Display for CouplingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObjectiveOnly => write!(f, "ObjOnly"),
            Self::ConstraintOnly => write!(f, "CstrOnly"),
            Self::Both => write!(f, "Both"),
            Self::None => write!(f, "None"),
        }
    }
}

impl CouplingType {
    pub fn has_objective_coupling(&self) -> bool {
        matches!(self, Self::ObjectiveOnly | Self::Both)
    }
    pub fn has_constraint_coupling(&self) -> bool {
        matches!(self, Self::ConstraintOnly | Self::Both)
    }
    pub fn combine(a: Self, b: Self) -> Self {
        let o = a.has_objective_coupling() || b.has_objective_coupling();
        let c = a.has_constraint_coupling() || b.has_constraint_coupling();
        match (o, c) {
            (true, true) => Self::Both,
            (true, false) => Self::ObjectiveOnly,
            (false, true) => Self::ConstraintOnly,
            _ => Self::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CqStatus {
    LICQ,
    MFCQ,
    Slater,
    NoneVerifiable,
    Unknown,
}

impl fmt::Display for CqStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LICQ => write!(f, "LICQ"),
            Self::MFCQ => write!(f, "MFCQ"),
            Self::Slater => write!(f, "Slater"),
            Self::NoneVerifiable => write!(f, "NoneVerifiable"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

impl CqStatus {
    pub fn is_sufficient_for_kkt(&self) -> bool {
        matches!(self, Self::LICQ | Self::MFCQ | Self::Slater)
    }
    pub fn strength(&self) -> u32 {
        match self {
            Self::LICQ => 3,
            Self::MFCQ => 2,
            Self::Slater => 1,
            _ => 0,
        }
    }
    pub fn implies(&self, other: &Self) -> bool {
        self.strength() >= other.strength()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureDimensions {
    pub n_x: usize,
    pub n_y: usize,
    pub m: usize,
    pub p: usize,
}

impl SignatureDimensions {
    pub fn new(n_x: usize, n_y: usize, m: usize, p: usize) -> Self {
        Self { n_x, n_y, m, p }
    }
    pub fn total_vars(&self) -> usize {
        self.n_x + self.n_y
    }
    pub fn total_constraints(&self) -> usize {
        self.m + self.p
    }
    pub fn is_small(&self) -> bool {
        self.total_vars() < 50
    }
    pub fn is_medium(&self) -> bool {
        let t = self.total_vars();
        t >= 50 && t < 500
    }
    pub fn is_large(&self) -> bool {
        self.total_vars() >= 500
    }
    pub fn size_category(&self) -> &'static str {
        if self.is_small() {
            "small"
        } else if self.is_medium() {
            "medium"
        } else {
            "large"
        }
    }
}

impl fmt::Display for SignatureDimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "(nx={},ny={},m={},p={})",
            self.n_x, self.n_y, self.m, self.p
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegerCounts {
    pub leader: usize,
    pub follower: usize,
}

impl IntegerCounts {
    pub fn new(leader: usize, follower: usize) -> Self {
        Self { leader, follower }
    }
    pub fn is_continuous(&self) -> bool {
        self.leader == 0 && self.follower == 0
    }
    pub fn has_integer_follower(&self) -> bool {
        self.follower > 0
    }
    pub fn has_integer_leader(&self) -> bool {
        self.leader > 0
    }
    pub fn total(&self) -> usize {
        self.leader + self.follower
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSignature {
    pub lower_type: LowerLevelType,
    pub var_types: IntegerCounts,
    pub cq_status: CqStatus,
    pub coupling: CouplingType,
    pub dims: SignatureDimensions,
}

impl ProblemSignature {
    pub fn new(
        lt: LowerLevelType,
        vt: IntegerCounts,
        cq: CqStatus,
        cp: CouplingType,
        d: SignatureDimensions,
    ) -> Self {
        Self {
            lower_type: lt,
            var_types: vt,
            cq_status: cq,
            coupling: cp,
            dims: d,
        }
    }
    pub fn is_blp(&self) -> bool {
        self.lower_type == LowerLevelType::LP && self.var_types.is_continuous()
    }
    pub fn is_miblp(&self) -> bool {
        !self.var_types.is_continuous()
    }
    pub fn kkt_applicable(&self) -> bool {
        self.lower_type.supports_kkt()
            && self.cq_status.is_sufficient_for_kkt()
            && !self.var_types.has_integer_follower()
    }
    pub fn strong_duality_applicable(&self) -> bool {
        self.lower_type.supports_strong_duality() && !self.var_types.has_integer_follower()
    }
    pub fn value_function_applicable(&self) -> bool {
        true
    }
    pub fn difficulty_score(&self) -> u32 {
        let mut s = self.lower_type.difficulty_score();
        if self.var_types.has_integer_leader() {
            s += 2;
        }
        if self.var_types.has_integer_follower() {
            s += 3;
        }
        if self.dims.is_large() {
            s += 2;
        }
        s
    }
    pub fn recommended_strategies(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.kkt_applicable() {
            v.push("KKT");
        }
        if self.strong_duality_applicable() {
            v.push("StrongDuality");
        }
        v.push("ValueFunction");
        if self.coupling.has_constraint_coupling() {
            v.push("CCG");
        }
        v
    }
}

impl fmt::Display for ProblemSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sig({},{},{},{},{})",
            self.lower_type,
            self.var_types.total(),
            self.cq_status,
            self.coupling,
            self.dims
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig() -> ProblemSignature {
        ProblemSignature::new(
            LowerLevelType::LP,
            IntegerCounts::new(0, 0),
            CqStatus::LICQ,
            CouplingType::Both,
            SignatureDimensions::new(10, 20, 15, 5),
        )
    }

    #[test]
    fn test_blp() {
        assert!(sig().is_blp());
    }
    #[test]
    fn test_kkt() {
        assert!(sig().kkt_applicable());
    }
    #[test]
    fn test_sd() {
        assert!(sig().strong_duality_applicable());
    }
    #[test]
    fn test_miblp() {
        let s = ProblemSignature::new(
            LowerLevelType::MixedInteger,
            IntegerCounts::new(5, 3),
            CqStatus::NoneVerifiable,
            CouplingType::Both,
            SignatureDimensions::new(10, 20, 15, 5),
        );
        assert!(s.is_miblp());
        assert!(!s.kkt_applicable());
    }
    #[test]
    fn test_strats() {
        assert!(sig().recommended_strategies().contains(&"KKT"));
    }
    #[test]
    fn test_difficulty() {
        assert!(sig().difficulty_score() < 10);
    }
    #[test]
    fn test_cq_implies() {
        assert!(CqStatus::LICQ.implies(&CqStatus::MFCQ));
    }
    #[test]
    fn test_coupling() {
        assert_eq!(
            CouplingType::combine(CouplingType::ObjectiveOnly, CouplingType::ConstraintOnly),
            CouplingType::Both
        );
    }
}

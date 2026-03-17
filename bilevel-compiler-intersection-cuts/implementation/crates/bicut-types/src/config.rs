use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceConfig {
    pub feasibility: f64,
    pub optimality: f64,
    pub integrality: f64,
    pub complementarity: f64,
    pub cut_violation: f64,
    pub numerical_zero: f64,
}

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            feasibility: 1e-6,
            optimality: 1e-6,
            integrality: 1e-5,
            complementarity: 1e-6,
            cut_violation: 1e-4,
            numerical_zero: 1e-10,
        }
    }
}

impl ToleranceConfig {
    pub fn strict() -> Self {
        Self {
            feasibility: 1e-8,
            optimality: 1e-8,
            integrality: 1e-7,
            complementarity: 1e-8,
            cut_violation: 1e-6,
            numerical_zero: 1e-12,
        }
    }
    pub fn relaxed() -> Self {
        Self {
            feasibility: 1e-4,
            optimality: 1e-4,
            integrality: 1e-3,
            complementarity: 1e-4,
            cut_violation: 1e-3,
            numerical_zero: 1e-8,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.feasibility <= 0.0 {
            return Err("feasibility tolerance must be positive".into());
        }
        if self.optimality <= 0.0 {
            return Err("optimality tolerance must be positive".into());
        }
        if self.integrality <= 0.0 {
            return Err("integrality tolerance must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverBackendType {
    Internal,
    Gurobi,
    Scip,
    HiGHS,
    Cplex,
}

impl fmt::Display for SolverBackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal => write!(f, "Internal"),
            Self::Gurobi => write!(f, "Gurobi"),
            Self::Scip => write!(f, "SCIP"),
            Self::HiGHS => write!(f, "HiGHS"),
            Self::Cplex => write!(f, "CPLEX"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub backend: SolverBackendType,
    pub time_limit_secs: f64,
    pub node_limit: u64,
    pub iteration_limit: u64,
    pub threads: usize,
    pub tolerances: ToleranceConfig,
    pub verbose: bool,
    pub presolve: bool,
    pub cuts_enabled: bool,
    pub heuristics_enabled: bool,
    pub mip_gap: f64,
    pub parameters: std::collections::HashMap<String, String>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            backend: SolverBackendType::Internal,
            time_limit_secs: 3600.0,
            node_limit: u64::MAX,
            iteration_limit: u64::MAX,
            threads: 1,
            tolerances: ToleranceConfig::default(),
            verbose: false,
            presolve: true,
            cuts_enabled: true,
            heuristics_enabled: true,
            mip_gap: 1e-4,
            parameters: std::collections::HashMap::new(),
        }
    }
}

impl SolverConfig {
    pub fn with_time_limit(mut self, secs: f64) -> Self {
        self.time_limit_secs = secs;
        self
    }
    pub fn with_threads(mut self, t: usize) -> Self {
        self.threads = t;
        self
    }
    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }
    pub fn with_backend(mut self, b: SolverBackendType) -> Self {
        self.backend = b;
        self
    }
    pub fn set_parameter(&mut self, key: &str, val: &str) {
        self.parameters.insert(key.into(), val.into());
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.time_limit_secs <= 0.0 {
            return Err("time limit must be positive".into());
        }
        if self.threads == 0 {
            return Err("threads must be >= 1".into());
        }
        if self.mip_gap < 0.0 {
            return Err("MIP gap must be non-negative".into());
        }
        self.tolerances.validate()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReformulationChoice {
    Auto,
    KKT,
    StrongDuality,
    ValueFunction,
    CCG,
}

impl fmt::Display for ReformulationChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "Auto"),
            Self::KKT => write!(f, "KKT"),
            Self::StrongDuality => write!(f, "StrongDuality"),
            Self::ValueFunction => write!(f, "ValueFunction"),
            Self::CCG => write!(f, "CCG"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplementarityEncoding {
    BigM,
    SOS1,
    Indicator,
    Auto,
}

impl fmt::Display for ComplementarityEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BigM => write!(f, "BigM"),
            Self::SOS1 => write!(f, "SOS1"),
            Self::Indicator => write!(f, "Indicator"),
            Self::Auto => write!(f, "Auto"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerConfig {
    pub reformulation: ReformulationChoice,
    pub complementarity_encoding: ComplementarityEncoding,
    pub big_m_method: String,
    pub big_m_default: f64,
    pub enable_preprocessing: bool,
    pub enable_bound_tightening: bool,
    pub certificate_generation: bool,
    pub spot_check_count: usize,
    pub output_format: String,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            reformulation: ReformulationChoice::Auto,
            complementarity_encoding: ComplementarityEncoding::Auto,
            big_m_method: "lp_tightening".into(),
            big_m_default: 1e6,
            enable_preprocessing: true,
            enable_bound_tightening: true,
            certificate_generation: true,
            spot_check_count: 10,
            output_format: "mps".into(),
        }
    }
}

impl CompilerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.big_m_default <= 0.0 {
            return Err("big-M default must be positive".into());
        }
        if !["mps", "lp", "json"].contains(&self.output_format.as_str()) {
            return Err(format!("Unknown output format: {}", self.output_format));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutConfig {
    pub enable_intersection_cuts: bool,
    pub enable_gomory_cuts: bool,
    pub enable_disjunctive_cuts: bool,
    pub enable_strengthening: bool,
    pub max_rounds: usize,
    pub max_cuts_per_round: usize,
    pub min_violation: f64,
    pub max_density: f64,
    pub aging_limit: usize,
    pub cache_size: usize,
    pub gap_threshold: f64,
}

impl Default for CutConfig {
    fn default() -> Self {
        Self {
            enable_intersection_cuts: true,
            enable_gomory_cuts: true,
            enable_disjunctive_cuts: false,
            enable_strengthening: true,
            max_rounds: 20,
            max_cuts_per_round: 50,
            min_violation: 1e-4,
            max_density: 0.5,
            aging_limit: 10,
            cache_size: 10000,
            gap_threshold: 0.01,
        }
    }
}

impl CutConfig {
    pub fn aggressive() -> Self {
        Self {
            max_rounds: 50,
            max_cuts_per_round: 100,
            min_violation: 1e-5,
            enable_disjunctive_cuts: true,
            ..Self::default()
        }
    }
    pub fn conservative() -> Self {
        Self {
            max_rounds: 5,
            max_cuts_per_round: 20,
            min_violation: 1e-3,
            ..Self::default()
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.max_rounds == 0 {
            return Err("max_rounds must be > 0".into());
        }
        if self.min_violation <= 0.0 {
            return Err("min_violation must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeSelectionStrategy {
    BestFirst,
    DepthFirst,
    BreadthFirst,
    Hybrid,
}

impl fmt::Display for NodeSelectionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BestFirst => write!(f, "BestFirst"),
            Self::DepthFirst => write!(f, "DepthFirst"),
            Self::BreadthFirst => write!(f, "BreadthFirst"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchingStrategy {
    MostFractional,
    StrongBranching,
    ReliabilityBranching,
    PseudocostBranching,
    Hybrid,
}

impl fmt::Display for BranchingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MostFractional => write!(f, "MostFractional"),
            Self::StrongBranching => write!(f, "StrongBranching"),
            Self::ReliabilityBranching => write!(f, "ReliabilityBranching"),
            Self::PseudocostBranching => write!(f, "Pseudocost"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchConfig {
    pub node_selection: NodeSelectionStrategy,
    pub branching_strategy: BranchingStrategy,
    pub strong_branching_candidates: usize,
    pub reliability_threshold: usize,
    pub enable_heuristics: bool,
    pub heuristic_frequency: usize,
    pub plunge_depth: usize,
}

impl Default for BranchConfig {
    fn default() -> Self {
        Self {
            node_selection: NodeSelectionStrategy::Hybrid,
            branching_strategy: BranchingStrategy::ReliabilityBranching,
            strong_branching_candidates: 10,
            reliability_threshold: 8,
            enable_heuristics: true,
            heuristic_frequency: 5,
            plunge_depth: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BicutConfig {
    pub solver: SolverConfig,
    pub compiler: CompilerConfig,
    pub cuts: CutConfig,
    pub branch: BranchConfig,
}

impl Default for BicutConfig {
    fn default() -> Self {
        Self {
            solver: SolverConfig::default(),
            compiler: CompilerConfig::default(),
            cuts: CutConfig::default(),
            branch: BranchConfig::default(),
        }
    }
}

impl BicutConfig {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if let Err(e) = self.solver.validate() {
            errors.push(format!("solver: {}", e));
        }
        if let Err(e) = self.compiler.validate() {
            errors.push(format!("compiler: {}", e));
        }
        if let Err(e) = self.cuts.validate() {
            errors.push(format!("cuts: {}", e));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let c = BicutConfig::default();
        assert!(c.validate().is_ok());
    }
    #[test]
    fn test_tolerance() {
        let t = ToleranceConfig::strict();
        assert!(t.feasibility < ToleranceConfig::default().feasibility);
    }
    #[test]
    fn test_solver_config() {
        let s = SolverConfig::default().with_time_limit(100.0);
        assert!((s.time_limit_secs - 100.0).abs() < 1e-10);
    }
    #[test]
    fn test_cut_aggressive() {
        let c = CutConfig::aggressive();
        assert!(c.max_rounds > CutConfig::default().max_rounds);
    }
    #[test]
    fn test_compiler_validate() {
        let c = CompilerConfig::default();
        assert!(c.validate().is_ok());
    }
    #[test]
    fn test_invalid_config() {
        let mut c = SolverConfig::default();
        c.time_limit_secs = -1.0;
        assert!(c.validate().is_err());
    }
    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", SolverBackendType::Gurobi), "Gurobi");
    }
}

//! Decomposition method types and configuration.
//!
//! Defines the four decomposition strategies (Benders, Dantzig-Wolfe,
//! Lagrangian relaxation, or none), along with result types, configuration
//! structs, and structure detection types.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The decomposition methods the oracle can recommend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecompositionMethod {
    Benders,
    DantzigWolfe,
    LagrangianRelaxation,
    None,
}

impl DecompositionMethod {
    pub fn all() -> &'static [DecompositionMethod] {
        &[
            DecompositionMethod::Benders,
            DecompositionMethod::DantzigWolfe,
            DecompositionMethod::LagrangianRelaxation,
            DecompositionMethod::None,
        ]
    }

    pub fn index(self) -> usize {
        match self {
            DecompositionMethod::Benders => 0,
            DecompositionMethod::DantzigWolfe => 1,
            DecompositionMethod::LagrangianRelaxation => 2,
            DecompositionMethod::None => 3,
        }
    }

    pub fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(DecompositionMethod::Benders),
            1 => Some(DecompositionMethod::DantzigWolfe),
            2 => Some(DecompositionMethod::LagrangianRelaxation),
            3 => Some(DecompositionMethod::None),
            _ => None,
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            DecompositionMethod::Benders => "BD",
            DecompositionMethod::DantzigWolfe => "DW",
            DecompositionMethod::LagrangianRelaxation => "LR",
            DecompositionMethod::None => "NONE",
        }
    }

    pub fn requires_partition(self) -> bool {
        matches!(
            self,
            DecompositionMethod::Benders
                | DecompositionMethod::DantzigWolfe
                | DecompositionMethod::LagrangianRelaxation
        )
    }

    pub fn class_count() -> usize {
        4
    }
}

impl fmt::Display for DecompositionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecompositionMethod::Benders => write!(f, "Benders Decomposition"),
            DecompositionMethod::DantzigWolfe => write!(f, "Dantzig-Wolfe Decomposition"),
            DecompositionMethod::LagrangianRelaxation => write!(f, "Lagrangian Relaxation"),
            DecompositionMethod::None => write!(f, "No Decomposition"),
        }
    }
}

/// Status of a decomposition run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecompositionStatus {
    Optimal,
    Feasible,
    Infeasible,
    Unbounded,
    TimeLimitReached,
    IterationLimitReached,
    NumericalDifficulty,
    Unknown,
}

impl fmt::Display for DecompositionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecompositionStatus::Optimal => write!(f, "Optimal"),
            DecompositionStatus::Feasible => write!(f, "Feasible"),
            DecompositionStatus::Infeasible => write!(f, "Infeasible"),
            DecompositionStatus::Unbounded => write!(f, "Unbounded"),
            DecompositionStatus::TimeLimitReached => write!(f, "Time limit"),
            DecompositionStatus::IterationLimitReached => write!(f, "Iteration limit"),
            DecompositionStatus::NumericalDifficulty => write!(f, "Numerical difficulty"),
            DecompositionStatus::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Result from running a decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompResult {
    pub method: DecompositionMethod,
    pub status: DecompositionStatus,
    pub dual_bound: f64,
    pub primal_bound: f64,
    pub gap: f64,
    pub time_seconds: f64,
    pub iterations: usize,
    pub num_cuts: usize,
    pub num_columns: usize,
    pub num_subproblems: usize,
    pub root_dual_bound: f64,
}

impl DecompResult {
    pub fn new(method: DecompositionMethod) -> Self {
        Self {
            method,
            status: DecompositionStatus::Unknown,
            dual_bound: f64::NEG_INFINITY,
            primal_bound: f64::INFINITY,
            gap: f64::INFINITY,
            time_seconds: 0.0,
            iterations: 0,
            num_cuts: 0,
            num_columns: 0,
            num_subproblems: 0,
            root_dual_bound: f64::NEG_INFINITY,
        }
    }

    pub fn is_optimal(&self) -> bool {
        self.status == DecompositionStatus::Optimal
    }

    pub fn relative_gap(&self) -> f64 {
        if self.primal_bound.abs() < 1e-10 {
            if self.dual_bound.abs() < 1e-10 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            ((self.primal_bound - self.dual_bound) / self.primal_bound.abs()).abs()
        }
    }

    pub fn closed_gap_fraction(&self) -> f64 {
        if self.root_dual_bound == f64::NEG_INFINITY {
            return 0.0;
        }
        let total_gap = self.primal_bound - self.root_dual_bound;
        if total_gap.abs() < 1e-10 {
            return 1.0;
        }
        let current_gap = self.primal_bound - self.dual_bound;
        1.0 - (current_gap / total_gap).max(0.0).min(1.0)
    }
}

impl fmt::Display for DecompResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}]: gap={:.4}%, time={:.2}s, iters={}",
            self.method,
            self.status,
            self.gap * 100.0,
            self.time_seconds,
            self.iterations
        )
    }
}

/// Configuration for decomposition runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionConfig {
    pub time_limit: f64,
    pub gap_tolerance: f64,
    pub max_iterations: usize,
    pub verbose: bool,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            time_limit: 3600.0,
            gap_tolerance: 1e-4,
            max_iterations: 10000,
            verbose: false,
        }
    }
}

/// Benders-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BendersConfig {
    pub base: DecompositionConfig,
    pub use_magnanti_wong: bool,
    pub use_callback: bool,
    pub max_cuts_per_iter: usize,
    pub cut_tolerance: f64,
    pub use_warm_start: bool,
    pub multi_cut: bool,
}

impl Default for BendersConfig {
    fn default() -> Self {
        Self {
            base: DecompositionConfig::default(),
            use_magnanti_wong: true,
            use_callback: true,
            max_cuts_per_iter: 100,
            cut_tolerance: 1e-6,
            use_warm_start: true,
            multi_cut: true,
        }
    }
}

/// Dantzig-Wolfe specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DWConfig {
    pub base: DecompositionConfig,
    pub pricing_strategy: PricingStrategy,
    pub stabilization: StabilizationType,
    pub max_columns_per_iter: usize,
    pub column_deletion_threshold: f64,
    pub branching: BranchingStrategy,
}

impl Default for DWConfig {
    fn default() -> Self {
        Self {
            base: DecompositionConfig::default(),
            pricing_strategy: PricingStrategy::ReducedCost,
            stabilization: StabilizationType::DuSmoothing { alpha: 0.3 },
            max_columns_per_iter: 50,
            column_deletion_threshold: 1e-8,
            branching: BranchingStrategy::RyanFoster,
        }
    }
}

/// Pricing strategy for column generation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PricingStrategy {
    ReducedCost,
    Farkas,
    Heuristic,
    Combined,
}

/// Stabilization method for column generation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StabilizationType {
    None,
    DuSmoothing { alpha: f64 },
    BoxStep { delta: f64 },
    Wentges { alpha: f64 },
}

/// Branching strategy for branch-and-price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BranchingStrategy {
    RyanFoster,
    Generic,
    Hybrid,
}

/// Lagrangian relaxation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagrangianConfig {
    pub base: DecompositionConfig,
    pub step_rule: StepRule,
    pub initial_step_size: f64,
    pub step_decay: f64,
    pub heuristic_frequency: usize,
    pub bundle_size: usize,
}

impl Default for LagrangianConfig {
    fn default() -> Self {
        Self {
            base: DecompositionConfig::default(),
            step_rule: StepRule::Polyak,
            initial_step_size: 2.0,
            step_decay: 0.95,
            heuristic_frequency: 10,
            bundle_size: 20,
        }
    }
}

/// Step size rule for subgradient method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StepRule {
    Polyak,
    Constant,
    Harmonic,
    GeometricDecay { ratio: f64 },
}

/// Structure types that can be detected in a MIP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructureType {
    BlockAngular,
    BorderedBlockDiagonal,
    Staircase,
    Network,
    Unstructured,
}

impl StructureType {
    pub fn suggested_methods(self) -> Vec<DecompositionMethod> {
        match self {
            StructureType::BlockAngular => {
                vec![DecompositionMethod::DantzigWolfe]
            }
            StructureType::BorderedBlockDiagonal => {
                vec![
                    DecompositionMethod::DantzigWolfe,
                    DecompositionMethod::LagrangianRelaxation,
                ]
            }
            StructureType::Staircase => {
                vec![
                    DecompositionMethod::Benders,
                    DecompositionMethod::LagrangianRelaxation,
                ]
            }
            StructureType::Network => {
                vec![
                    DecompositionMethod::LagrangianRelaxation,
                    DecompositionMethod::DantzigWolfe,
                ]
            }
            StructureType::Unstructured => {
                vec![DecompositionMethod::None]
            }
        }
    }
}

impl fmt::Display for StructureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StructureType::BlockAngular => write!(f, "Block Angular"),
            StructureType::BorderedBlockDiagonal => write!(f, "Bordered Block Diagonal"),
            StructureType::Staircase => write!(f, "Staircase"),
            StructureType::Network => write!(f, "Network"),
            StructureType::Unstructured => write!(f, "Unstructured"),
        }
    }
}

/// Result of structure detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub structure_type: StructureType,
    pub confidence: f64,
    pub num_blocks: usize,
    pub linking_constraints: usize,
    pub linking_variables: usize,
    pub block_sizes: Vec<usize>,
    pub detection_time_seconds: f64,
}

impl DetectionResult {
    pub fn unstructured() -> Self {
        Self {
            structure_type: StructureType::Unstructured,
            confidence: 1.0,
            num_blocks: 1,
            linking_constraints: 0,
            linking_variables: 0,
            block_sizes: Vec::new(),
            detection_time_seconds: 0.0,
        }
    }

    pub fn max_block_size(&self) -> usize {
        self.block_sizes.iter().copied().max().unwrap_or(0)
    }

    pub fn min_block_size(&self) -> usize {
        self.block_sizes.iter().copied().min().unwrap_or(0)
    }

    pub fn balance_ratio(&self) -> f64 {
        if self.block_sizes.is_empty() {
            return 1.0;
        }
        let max = self.max_block_size() as f64;
        let min = self.min_block_size() as f64;
        if max < 1e-15 {
            1.0
        } else {
            min / max
        }
    }

    pub fn is_well_structured(&self) -> bool {
        self.structure_type != StructureType::Unstructured
            && self.confidence >= 0.5
            && self.num_blocks >= 2
    }
}

/// Comparison of two decomposition results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompComparison {
    pub method_a: DecompositionMethod,
    pub method_b: DecompositionMethod,
    pub speedup: f64,
    pub gap_improvement: f64,
    pub winner: DecompositionMethod,
}

impl DecompComparison {
    pub fn compare(a: &DecompResult, b: &DecompResult) -> Self {
        let speedup = if b.time_seconds > 1e-10 {
            b.time_seconds / a.time_seconds.max(1e-10)
        } else {
            1.0
        };
        let gap_improvement = if a.gap > 1e-10 {
            (b.gap - a.gap) / a.gap
        } else {
            0.0
        };
        let winner = if a.gap < b.gap - 1e-10 {
            a.method
        } else if b.gap < a.gap - 1e-10 {
            b.method
        } else if a.time_seconds < b.time_seconds {
            a.method
        } else {
            b.method
        };
        Self {
            method_a: a.method,
            method_b: b.method,
            speedup,
            gap_improvement,
            winner,
        }
    }
}

/// Oracle prediction with confidence scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OraclePrediction {
    pub recommended: DecompositionMethod,
    pub confidence: f64,
    pub class_probabilities: [f64; 4],
    pub margin: f64,
    pub feature_importances: Vec<(String, f64)>,
}

impl OraclePrediction {
    pub fn new(probs: [f64; 4]) -> Self {
        let mut best_idx = 0;
        let mut best_val = probs[0];
        let mut second_val = f64::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_val {
                second_val = best_val;
                best_val = p;
                best_idx = i;
            } else if p > second_val {
                second_val = p;
            }
        }
        let margin = best_val - second_val.max(0.0);
        Self {
            recommended: DecompositionMethod::from_index(best_idx)
                .unwrap_or(DecompositionMethod::None),
            confidence: best_val,
            class_probabilities: probs,
            margin,
            feature_importances: Vec::new(),
        }
    }

    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold && self.margin >= threshold * 0.5
    }

    pub fn top_k(&self, k: usize) -> Vec<(DecompositionMethod, f64)> {
        let mut indexed: Vec<(usize, f64)> =
            self.class_probabilities.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
            .into_iter()
            .take(k)
            .filter_map(|(i, p)| DecompositionMethod::from_index(i).map(|m| (m, p)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_index_roundtrip() {
        for m in DecompositionMethod::all() {
            assert_eq!(DecompositionMethod::from_index(m.index()), Some(*m));
        }
    }

    #[test]
    fn test_method_display() {
        assert!(DecompositionMethod::Benders.to_string().contains("Benders"));
    }

    #[test]
    fn test_decomp_result_relative_gap() {
        let mut r = DecompResult::new(DecompositionMethod::Benders);
        r.dual_bound = 90.0;
        r.primal_bound = 100.0;
        assert!((r.relative_gap() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_decomp_result_closed_gap() {
        let mut r = DecompResult::new(DecompositionMethod::DantzigWolfe);
        r.root_dual_bound = 80.0;
        r.dual_bound = 95.0;
        r.primal_bound = 100.0;
        let frac = r.closed_gap_fraction();
        assert!(frac > 0.7 && frac < 0.8);
    }

    #[test]
    fn test_detection_result_balance() {
        let dr = DetectionResult {
            structure_type: StructureType::BlockAngular,
            confidence: 0.9,
            num_blocks: 3,
            linking_constraints: 5,
            linking_variables: 0,
            block_sizes: vec![10, 20, 15],
            detection_time_seconds: 0.1,
        };
        assert!((dr.balance_ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_detection_result_well_structured() {
        let dr = DetectionResult {
            structure_type: StructureType::BlockAngular,
            confidence: 0.9,
            num_blocks: 3,
            linking_constraints: 5,
            linking_variables: 0,
            block_sizes: vec![10, 20, 15],
            detection_time_seconds: 0.1,
        };
        assert!(dr.is_well_structured());
    }

    #[test]
    fn test_structure_suggested_methods() {
        let methods = StructureType::BlockAngular.suggested_methods();
        assert!(methods.contains(&DecompositionMethod::DantzigWolfe));
    }

    #[test]
    fn test_oracle_prediction() {
        let pred = OraclePrediction::new([0.1, 0.6, 0.2, 0.1]);
        assert_eq!(pred.recommended, DecompositionMethod::DantzigWolfe);
        assert!((pred.confidence - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_oracle_prediction_top_k() {
        let pred = OraclePrediction::new([0.1, 0.6, 0.2, 0.1]);
        let top2 = pred.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, DecompositionMethod::DantzigWolfe);
    }

    #[test]
    fn test_decomp_comparison() {
        let mut a = DecompResult::new(DecompositionMethod::Benders);
        a.gap = 0.01;
        a.time_seconds = 10.0;
        let mut b = DecompResult::new(DecompositionMethod::DantzigWolfe);
        b.gap = 0.05;
        b.time_seconds = 20.0;
        let cmp = DecompComparison::compare(&a, &b);
        assert_eq!(cmp.winner, DecompositionMethod::Benders);
    }

    #[test]
    fn test_benders_config_default() {
        let cfg = BendersConfig::default();
        assert!(cfg.use_magnanti_wong);
        assert_eq!(cfg.max_cuts_per_iter, 100);
    }

    #[test]
    fn test_dw_config_default() {
        let cfg = DWConfig::default();
        assert_eq!(cfg.max_columns_per_iter, 50);
    }

    #[test]
    fn test_lagrangian_config_default() {
        let cfg = LagrangianConfig::default();
        assert_eq!(cfg.initial_step_size, 2.0);
    }

    #[test]
    fn test_decomp_status_display() {
        assert_eq!(DecompositionStatus::Optimal.to_string(), "Optimal");
    }

    #[test]
    fn test_method_requires_partition() {
        assert!(DecompositionMethod::Benders.requires_partition());
        assert!(!DecompositionMethod::None.requires_partition());
    }
}

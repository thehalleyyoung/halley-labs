//! Repair synthesis for Penumbra floating-point error remediation.

use eag_builder::NodeIndex;
use expression_rewrite::Expr;
use penumbra_types::FpOperation;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single repair action targeting a node in the EAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairAction {
    pub id: Uuid,
    pub target_node: NodeIndex,
    pub original_expr: Expr,
    pub repaired_expr: Expr,
    pub original_operation: FpOperation,
    pub expected_improvement: f64,
    pub description: String,
    pub strategy: RepairStrategy,
}

impl RepairAction {
    pub fn new(
        target_node: NodeIndex,
        original_expr: Expr,
        repaired_expr: Expr,
        original_operation: FpOperation,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            target_node,
            original_expr,
            repaired_expr,
            original_operation,
            expected_improvement: 0.0,
            description: String::new(),
            strategy: RepairStrategy::ExpressionRewrite,
        }
    }
}

/// Strategy used to synthesize a repair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepairStrategy {
    ExpressionRewrite,
    PrecisionPromotion,
    CompensatedAlgorithm,
    AlgorithmSubstitution,
    Custom(String),
}

/// A plan consisting of multiple coordinated repair actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairPlan {
    pub id: Uuid,
    pub repairs: Vec<RepairAction>,
    pub total_expected_improvement: f64,
    pub description: String,
}

impl RepairPlan {
    pub fn new(repairs: Vec<RepairAction>) -> Self {
        let total = repairs.iter().map(|r| r.expected_improvement).sum();
        Self {
            id: Uuid::new_v4(),
            repairs,
            total_expected_improvement: total,
            description: String::new(),
        }
    }
}

// Evaluation submodule: cross-validation, ablation, metrics, and hypothesis testing.

pub mod cross_validation;
pub mod ablation;
pub mod metrics;
pub mod hypothesis;

pub use cross_validation::{NestedCV, StratifiedKFold, CVResults};
pub use ablation::{AblationStudy, AblationConfig, FeatureSetConfig};
pub use metrics::{compute_mcnemar, holm_bonferroni, bootstrap_ci, spearman_correlation};
pub use hypothesis::{HypothesisHarness, HypothesisResult, HypothesisOutcome};

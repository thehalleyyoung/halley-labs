//! Problem classification for bilevel optimization.
//!
//! Categorizes bilevel problems by structure (linear, convex, nonconvex),
//! estimates difficulty, and extracts instance features for benchmarking.

use bicut_types::{
    BilevelProblem, CouplingType, DifficultyClass, LowerLevelType, ProblemSignature, SparseMatrix,
    SparseMatrixCsr, DEFAULT_TOLERANCE,
};
use log::debug;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Structural category of a bilevel problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructuralCategory {
    /// Both levels are linear.
    LinearLinear,
    /// Linear upper, quadratic lower.
    LinearQuadratic,
    /// Linear upper, mixed-integer lower.
    LinearMixedInteger,
    /// Quadratic upper, linear lower.
    QuadraticLinear,
    /// General convex.
    Convex,
    /// General nonconvex.
    Nonconvex,
}

/// Comprehensive classification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    pub category: StructuralCategory,
    pub difficulty: DifficultyClass,
    pub difficulty_score: f64,
    pub features: InstanceFeatures,
    pub signature: ProblemSignature,
    pub benchmark_tags: Vec<String>,
    pub notes: Vec<String>,
}

/// Numerical features of a problem instance, useful for ML-based benchmarking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceFeatures {
    pub num_leader_vars: usize,
    pub num_follower_vars: usize,
    pub num_upper_constraints: usize,
    pub num_lower_constraints: usize,
    pub var_ratio: f64,
    pub constraint_ratio: f64,
    pub lower_density: f64,
    pub linking_density: f64,
    pub upper_density: f64,
    pub coefficient_range: f64,
    pub obj_sparsity: f64,
    pub coupling_strength: f64,
    pub has_linking: bool,
    pub has_upper_constraints: bool,
    pub instance_hash: String,
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

/// Configuration for problem classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub tolerance: f64,
    pub large_problem_threshold: usize,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            large_problem_threshold: 500,
        }
    }
}

/// Problem classification engine.
pub struct ProblemClassifier {
    config: ClassifierConfig,
}

impl ProblemClassifier {
    pub fn new(config: ClassifierConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ClassifierConfig::default())
    }

    /// Perform complete classification of a bilevel problem.
    pub fn classify(
        &self,
        problem: &BilevelProblem,
        sig: &ProblemSignature,
    ) -> ClassificationReport {
        let category = self.determine_category(sig);
        let features = self.extract_features(problem, sig);
        let difficulty_score = self.compute_difficulty_score(&features, sig);
        let difficulty = score_to_class(difficulty_score);
        let benchmark_tags = self.generate_tags(sig, &category, &features);
        let notes = self.generate_notes(sig, &category, &features);

        debug!(
            "Classified problem: {:?}, difficulty={:?} ({:.2})",
            category, difficulty, difficulty_score
        );

        ClassificationReport {
            category,
            difficulty,
            difficulty_score,
            features,
            signature: sig.clone(),
            benchmark_tags,
            notes,
        }
    }

    /// Determine the structural category.
    pub fn determine_category(&self, sig: &ProblemSignature) -> StructuralCategory {
        match sig.lower_type {
            LowerLevelType::LP => {
                if sig.has_integer_upper {
                    StructuralCategory::LinearMixedInteger
                } else {
                    StructuralCategory::LinearLinear
                }
            }
            LowerLevelType::QP => {
                if sig.has_integer_upper {
                    StructuralCategory::Convex
                } else {
                    StructuralCategory::LinearQuadratic
                }
            }
            LowerLevelType::MILP | LowerLevelType::MIQP => StructuralCategory::LinearMixedInteger,
            LowerLevelType::ConvexNLP => StructuralCategory::Convex,
            LowerLevelType::GeneralNLP => StructuralCategory::Nonconvex,
        }
    }

    /// Extract numerical instance features.
    pub fn extract_features(
        &self,
        problem: &BilevelProblem,
        sig: &ProblemSignature,
    ) -> InstanceFeatures {
        let nx = sig.num_leader_vars;
        let ny = sig.num_follower_vars;
        let mu = sig.num_upper_constraints;
        let ml = sig.num_lower_constraints;

        let var_ratio = if ny > 0 { nx as f64 / ny as f64 } else { 0.0 };
        let constraint_ratio = if ml > 0 { mu as f64 / ml as f64 } else { 0.0 };

        let lower_total = (ml * ny).max(1);
        let lower_nnz = problem.lower_a.entries.len();
        let lower_density = lower_nnz as f64 / lower_total as f64;

        let linking_total = (problem.lower_linking_b.rows * problem.lower_linking_b.cols).max(1);
        let linking_nnz = problem.lower_linking_b.entries.len();
        let linking_density = linking_nnz as f64 / linking_total as f64;

        let upper_total =
            (problem.upper_constraints_a.rows * problem.upper_constraints_a.cols).max(1);
        let upper_nnz = problem.upper_constraints_a.entries.len();
        let upper_density = upper_nnz as f64 / upper_total as f64;

        let coefficient_range = compute_coefficient_range(&problem.lower_a);

        let obj_total = problem.lower_obj_c.len().max(1);
        let obj_nnz = problem
            .lower_obj_c
            .iter()
            .filter(|&&c| c.abs() > self.config.tolerance)
            .count();
        let obj_sparsity = 1.0 - (obj_nnz as f64 / obj_total as f64);

        let coupling_strength = compute_coupling_strength(problem);

        let hash = compute_instance_hash(problem);

        InstanceFeatures {
            num_leader_vars: nx,
            num_follower_vars: ny,
            num_upper_constraints: mu,
            num_lower_constraints: ml,
            var_ratio,
            constraint_ratio,
            lower_density,
            linking_density,
            upper_density,
            coefficient_range,
            obj_sparsity,
            coupling_strength,
            has_linking: linking_nnz > 0,
            has_upper_constraints: mu > 0,
            instance_hash: hash,
        }
    }

    /// Compute a difficulty score in [0, 1].
    pub fn compute_difficulty_score(
        &self,
        features: &InstanceFeatures,
        sig: &ProblemSignature,
    ) -> f64 {
        let mut score = 0.0;

        // Size contribution (log scale)
        let size = (sig.total_vars() + sig.total_constraints()) as f64;
        let size_score = (size.ln() / 15.0).min(1.0);
        score += 0.25 * size_score;

        // Lower-level type contribution
        let type_score = match sig.lower_type {
            LowerLevelType::LP => 0.1,
            LowerLevelType::QP => 0.3,
            LowerLevelType::MILP => 0.6,
            LowerLevelType::MIQP => 0.7,
            LowerLevelType::ConvexNLP => 0.5,
            LowerLevelType::GeneralNLP => 0.9,
        };
        score += 0.3 * type_score;

        // Integer variables contribution
        if sig.has_integer_upper {
            score += 0.1;
        }
        if sig.has_integer_lower {
            score += 0.15;
        }

        // Coupling contribution
        let coupling_score = match sig.coupling_type {
            CouplingType::None => 0.0,
            CouplingType::ObjectiveOnly => 0.2,
            CouplingType::ConstraintOnly => 0.4,
            CouplingType::Both => 0.6,
        };
        score += 0.15 * coupling_score;

        // Density contribution (denser = harder)
        score += 0.05 * features.lower_density;

        // Coefficient range (larger range = harder numerics)
        let range_score = if features.coefficient_range > 1e6 {
            0.8
        } else if features.coefficient_range > 1e3 {
            0.4
        } else {
            0.1
        };
        score += 0.1 * range_score;

        score.min(1.0)
    }

    /// Generate benchmark tags for the problem.
    fn generate_tags(
        &self,
        sig: &ProblemSignature,
        category: &StructuralCategory,
        features: &InstanceFeatures,
    ) -> Vec<String> {
        let mut tags = Vec::new();

        tags.push(format!("{:?}", category));
        tags.push(format!("lower-{}", sig.lower_type));
        tags.push(format!("coupling-{}", sig.coupling_type));

        if sig.has_integers() {
            tags.push("integer".to_string());
        }
        if sig.total_vars() > self.config.large_problem_threshold {
            tags.push("large".to_string());
        } else if sig.total_vars() < 20 {
            tags.push("small".to_string());
        } else {
            tags.push("medium".to_string());
        }

        if features.lower_density > 0.5 {
            tags.push("dense".to_string());
        } else if features.lower_density < 0.1 {
            tags.push("sparse".to_string());
        }

        if features.coupling_strength > 0.5 {
            tags.push("strongly-coupled".to_string());
        } else if features.coupling_strength < 0.1 {
            tags.push("weakly-coupled".to_string());
        }

        tags
    }

    /// Generate classification notes.
    fn generate_notes(
        &self,
        sig: &ProblemSignature,
        category: &StructuralCategory,
        features: &InstanceFeatures,
    ) -> Vec<String> {
        let mut notes = Vec::new();

        if matches!(category, StructuralCategory::LinearLinear) {
            notes.push(
                "Linear-linear bilevel: KKT or strong duality reformulation recommended"
                    .to_string(),
            );
        }

        if features.var_ratio > 10.0 {
            notes.push(format!(
                "Leader has {:.0}x more variables than follower; decomposition may help",
                features.var_ratio
            ));
        } else if features.var_ratio < 0.1 {
            notes.push(
                "Follower has many more variables; Benders decomposition may be efficient"
                    .to_string(),
            );
        }

        if features.coefficient_range > 1e4 {
            notes.push("Large coefficient range detected; consider scaling".to_string());
        }

        if !features.has_linking {
            notes.push("No linking constraints; lower level is independent of leader".to_string());
        }

        if !features.has_upper_constraints {
            notes.push("No upper-level constraints; unconstrained leader".to_string());
        }

        notes
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn score_to_class(score: f64) -> DifficultyClass {
    if score < 0.15 {
        DifficultyClass::Easy
    } else if score < 0.35 {
        DifficultyClass::Moderate
    } else if score < 0.55 {
        DifficultyClass::Hard
    } else if score < 0.75 {
        DifficultyClass::VeryHard
    } else {
        DifficultyClass::Intractable
    }
}

fn compute_coefficient_range(sm: &SparseMatrix) -> f64 {
    let abs_vals: Vec<f64> = sm
        .entries
        .iter()
        .map(|e| e.value.abs())
        .filter(|&v| v > DEFAULT_TOLERANCE)
        .collect();
    if abs_vals.is_empty() {
        return 1.0;
    }
    let min_val = abs_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = abs_vals.iter().cloned().fold(0.0f64, f64::max);
    if min_val < DEFAULT_TOLERANCE {
        return f64::INFINITY;
    }
    max_val / min_val
}

fn compute_coupling_strength(problem: &BilevelProblem) -> f64 {
    let nx = problem.num_upper_vars;
    let ml = problem.num_lower_constraints;
    if nx == 0 || ml == 0 {
        return 0.0;
    }

    let total = (nx * ml) as f64;
    let nnz = problem
        .lower_linking_b
        .entries
        .iter()
        .filter(|e| e.value.abs() > DEFAULT_TOLERANCE)
        .count() as f64;
    nnz / total
}

fn compute_instance_hash(problem: &BilevelProblem) -> String {
    let mut hasher = Sha256::new();
    hasher.update(
        format!(
            "{},{},{},{}",
            problem.num_upper_vars,
            problem.num_lower_vars,
            problem.num_upper_constraints,
            problem.num_lower_constraints
        )
        .as_bytes(),
    );

    for &c in &problem.lower_obj_c {
        hasher.update(c.to_le_bytes());
    }
    for &b in &problem.lower_b {
        hasher.update(b.to_le_bytes());
    }
    for entry in &problem.lower_a.entries {
        hasher.update(entry.row.to_le_bytes());
        hasher.update(entry.col.to_le_bytes());
        hasher.update(entry.value.to_le_bytes());
    }

    hex::encode(hasher.finalize())[..16].to_string()
}

/// Classify difficulty of a specific problem signature without full features.
pub fn quick_difficulty(sig: &ProblemSignature) -> DifficultyClass {
    let size = sig.total_vars() + sig.total_constraints();
    let type_factor = match sig.lower_type {
        LowerLevelType::LP => 1.0,
        LowerLevelType::QP => 2.0,
        LowerLevelType::MILP => 4.0,
        LowerLevelType::MIQP => 5.0,
        LowerLevelType::ConvexNLP => 3.0,
        LowerLevelType::GeneralNLP => 8.0,
    };
    let int_factor = if sig.has_integers() { 2.0 } else { 1.0 };
    let adjusted = size as f64 * type_factor * int_factor;

    if adjusted < 50.0 {
        DifficultyClass::Easy
    } else if adjusted < 500.0 {
        DifficultyClass::Moderate
    } else if adjusted < 5000.0 {
        DifficultyClass::Hard
    } else if adjusted < 50000.0 {
        DifficultyClass::VeryHard
    } else {
        DifficultyClass::Intractable
    }
}

/// Compare two problem instances for benchmark similarity.
pub fn similarity_score(a: &InstanceFeatures, b: &InstanceFeatures) -> f64 {
    let mut score = 0.0;
    let mut weight = 0.0;

    // Size similarity
    let size_a = (a.num_leader_vars + a.num_follower_vars) as f64;
    let size_b = (b.num_leader_vars + b.num_follower_vars) as f64;
    let size_sim = 1.0 - ((size_a - size_b).abs() / (size_a + size_b + 1.0));
    score += 0.3 * size_sim;
    weight += 0.3;

    // Density similarity
    let dens_sim = 1.0 - (a.lower_density - b.lower_density).abs();
    score += 0.2 * dens_sim;
    weight += 0.2;

    // Coupling similarity
    let coup_sim = 1.0 - (a.coupling_strength - b.coupling_strength).abs();
    score += 0.2 * coup_sim;
    weight += 0.2;

    // Ratio similarity
    let ratio_sim = 1.0 - (a.var_ratio - b.var_ratio).abs().min(1.0);
    score += 0.15 * ratio_sim;
    weight += 0.15;

    // Coefficient range similarity (log scale)
    let log_range_a = a.coefficient_range.ln().max(0.0);
    let log_range_b = b.coefficient_range.ln().max(0.0);
    let range_sim = 1.0 - ((log_range_a - log_range_b).abs() / (log_range_a + log_range_b + 1.0));
    score += 0.15 * range_sim;
    weight += 0.15;

    score / weight
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::{BilevelProblem, SparseMatrix};

    fn make_lp_sig() -> ProblemSignature {
        ProblemSignature {
            lower_type: LowerLevelType::LP,
            coupling_type: CouplingType::Both,
            num_leader_vars: 3,
            num_follower_vars: 5,
            num_upper_constraints: 2,
            num_lower_constraints: 4,
            num_coupling_constraints: 1,
            has_integer_upper: false,
            has_integer_lower: false,
        }
    }

    fn make_test_problem() -> BilevelProblem {
        let mut lower_a = SparseMatrix::new(2, 3);
        lower_a.add_entry(0, 0, 1.0);
        lower_a.add_entry(1, 1, 1.0);

        BilevelProblem {
            upper_obj_c_x: vec![1.0, 0.0],
            upper_obj_c_y: vec![0.0, 1.0, 0.0],
            lower_obj_c: vec![1.0, 1.0, 0.0],
            lower_a,
            lower_b: vec![5.0, 5.0],
            lower_linking_b: SparseMatrix::new(2, 2),
            upper_constraints_a: SparseMatrix::new(1, 5),
            upper_constraints_b: vec![10.0],
            num_upper_vars: 2,
            num_lower_vars: 3,
            num_lower_constraints: 2,
            num_upper_constraints: 1,
        }
    }

    #[test]
    fn test_category_linear_linear() {
        let sig = make_lp_sig();
        let c = ProblemClassifier::with_defaults();
        assert_eq!(c.determine_category(&sig), StructuralCategory::LinearLinear);
    }

    #[test]
    fn test_category_milp() {
        let mut sig = make_lp_sig();
        sig.lower_type = LowerLevelType::MILP;
        let c = ProblemClassifier::with_defaults();
        assert_eq!(
            c.determine_category(&sig),
            StructuralCategory::LinearMixedInteger
        );
    }

    #[test]
    fn test_difficulty_score() {
        let sig = make_lp_sig();
        let p = make_test_problem();
        let c = ProblemClassifier::with_defaults();
        let features = c.extract_features(&p, &sig);
        let score = c.compute_difficulty_score(&features, &sig);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_full_classification() {
        let sig = make_lp_sig();
        let p = make_test_problem();
        let c = ProblemClassifier::with_defaults();
        let report = c.classify(&p, &sig);
        assert_eq!(report.category, StructuralCategory::LinearLinear);
        assert!(!report.benchmark_tags.is_empty());
    }

    #[test]
    fn test_instance_features() {
        let sig = make_lp_sig();
        let p = make_test_problem();
        let c = ProblemClassifier::with_defaults();
        let features = c.extract_features(&p, &sig);
        assert!(features.instance_hash.len() > 0);
        assert_eq!(features.num_leader_vars, 3);
    }

    #[test]
    fn test_quick_difficulty() {
        let sig = make_lp_sig();
        let d = quick_difficulty(&sig);
        assert!(matches!(
            d,
            DifficultyClass::Easy | DifficultyClass::Moderate
        ));
    }

    #[test]
    fn test_similarity_score() {
        let sig = make_lp_sig();
        let p = make_test_problem();
        let c = ProblemClassifier::with_defaults();
        let f1 = c.extract_features(&p, &sig);
        let f2 = c.extract_features(&p, &sig);
        let sim = similarity_score(&f1, &f2);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tags_include_size() {
        let sig = make_lp_sig();
        let p = make_test_problem();
        let c = ProblemClassifier::with_defaults();
        let report = c.classify(&p, &sig);
        assert!(report
            .benchmark_tags
            .iter()
            .any(|t| t == "small" || t == "medium"));
    }

    #[test]
    fn test_difficulty_increases_with_integers() {
        let sig1 = make_lp_sig();
        let mut sig2 = make_lp_sig();
        sig2.has_integer_lower = true;
        sig2.lower_type = LowerLevelType::MILP;

        let p = make_test_problem();
        let c = ProblemClassifier::with_defaults();
        let f1 = c.extract_features(&p, &sig1);
        let f2 = c.extract_features(&p, &sig2);
        let s1 = c.compute_difficulty_score(&f1, &sig1);
        let s2 = c.compute_difficulty_score(&f2, &sig2);
        assert!(s2 > s1);
    }
}
